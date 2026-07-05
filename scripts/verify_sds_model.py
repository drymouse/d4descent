"""
SDS(Score Distillation Sampling)で使う学習済みStable Diffusionモデルが、実行環境で正しく
ロードでき動作するかを検証するスクリプト。**GPUマシン上での実行を想定**（fp16を使うため。
CPUはfp16非対応/非現実的に遅いのでこのスクリプトでは対象外)。

2段階の検証:
  1. (既定) 生の StableDiffusion クラス(third_party/sds.py)を直接ロードし、テキスト埋め込みと
     compute_sds_loss が1回通ることを確認する。
  2. (--integration) 実際に URGrammar + URSDSTask(既存の動作実績があるSDSタスク)を使って、
     render01 -> SDS損失 -> 勾配 -> 書き換え、という本番と同じパイプラインを数ステップ回して
     配線を確認する。City文法へSDSを移植する前の「土台が動くか」の最終チェック。

使い方:
    # 元のモデル(要ライセンス同意 / アクセス権)を試す
    uv run python scripts/verify_sds_model.py --model stabilityai/stable-diffusion-2-1-base

    # アクセスできない場合の代替モデル
    uv run python scripts/verify_sds_model.py --model stable-diffusion-v1-5/stable-diffusion-v1-5

    # 実際のURSDSTaskパイプラインまで通して確認(数分かかる)
    uv run python scripts/verify_sds_model.py --model stable-diffusion-v1-5/stable-diffusion-v1-5 --integration
"""

import argparse
import math
import time

import torch


def check_cuda() -> None:
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)}")
        print(f"  total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: CUDA無し。fp16での本検証は失敗する可能性が高い。")


def run_quick_check(model_name: str, prompt: str, half_precision: bool) -> None:
    from d4descent.third_party.sds import StableDiffusion, SdConfig

    print(f"\n=== Tier 1: 生のStableDiffusionクラスの検証 (model={model_name}) ===")
    t0 = time.time()
    sd = StableDiffusion(SdConfig(pretrained_model_name_or_path=model_name, half_precision_weights=half_precision))
    print(f"モデルロード成功: {time.time() - t0:.1f}s, device={sd.device}, dtype={sd.weights_dtype}")

    t0 = time.time()
    emb = sd.get_text_embeds(prompt)
    print(f"テキスト埋め込みOK: shape={tuple(emb.shape)}, {time.time() - t0:.1f}s")

    t0 = time.time()
    img = torch.rand(1, 1, 512, 512, device=sd.device)
    text_embedding = torch.cat([sd.get_text_embeds(prompt), sd.get_text_embeds("")])
    loss = sd.compute_sds_loss(img, text_embedding)
    print(f"compute_sds_loss OK: loss={loss.item():.4f}, {time.time() - t0:.1f}s")
    print("Tier 1: ALL OK")


def run_integration_check(model_name: str, prompt: str, half_precision: bool, n_steps: int) -> None:
    """既存のUR文法 + URSDSTask(既存実装。CitySDSTaskの移植元テンプレート)で、
    実際の最適化ループ(render01 -> SDS損失 -> 勾配 -> 書き換え)を数ステップ回す。"""
    from d4descent.tasks._base import RenderArgs
    from d4descent.tasks.ur import URArgs
    from d4descent.losses.sds import SDSLossArgs
    from d4descent.third_party.sds import SdConfig
    from d4descent.optimizer import OptimizeArgs, optimize

    print(f"\n=== Tier 2: URSDSTaskでの実パイプライン検証 (model={model_name}, n_steps={n_steps}) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    render_args = RenderArgs(size=512, lim=(-1.5, 1.5), center_pixel=True, blur=1 / math.sqrt(2))
    loss_args = SDSLossArgs(
        prompt=prompt, neg_prompt="", sd=SdConfig(pretrained_model_name_or_path=model_name, half_precision_weights=half_precision)
    )
    ur_args = URArgs()

    t0 = time.time()
    task = ur_args.create(render_args, loss_args, device, None)
    print(f"URSDSTask構築(SDロード込み)成功: {time.time() - t0:.1f}s")

    optim_args = OptimizeArgs(
        n_steps=n_steps,
        scheduler="none",
        lr=0.01,
        proposal_trigger="step",
        propose_every=max(n_steps, 2),  # 検証用: 途中で書き換えは起こさず連続最適化のみ確認する
        proposal_size=8,
        batch_size=2,
        stopping_patience=None,
    )
    t0 = time.time()
    top_shape, loss, _all_objects, _all_metrics = optimize(task, optim_args, None, disable=False)
    print(f"optimize() 完走: {time.time() - t0:.1f}s, final_loss={loss:.4f}")
    print("Tier 2: ALL OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--prompt", type=str, default="street map of a city")
    parser.add_argument("--no_fp16", action="store_true", help="fp16を使わずfp32でロードする(通常は不要)")
    parser.add_argument("--integration", action="store_true", help="URSDSTaskでの実パイプライン検証まで行う")
    parser.add_argument("--n_steps", type=int, default=10, help="--integration時に回す連続最適化のステップ数")
    args = parser.parse_args()

    check_cuda()
    run_quick_check(args.model, args.prompt, half_precision=not args.no_fp16)
    if args.integration:
        run_integration_check(args.model, args.prompt, half_precision=not args.no_fp16, n_steps=args.n_steps)

    print("\n検証完了。問題なければ次のステップ(CitySDSTaskの実装)に進んでください。")


if __name__ == "__main__":
    main()
