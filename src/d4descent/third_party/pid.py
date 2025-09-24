from typing import Optional


class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        setpoint: float,
        error: Optional[float] = None,
        integral: Optional[float] = None,
        last_value: Optional[float] = None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.error = error
        self.integral = 0.0 if integral is None else integral
        self.last_value = last_value

    def update(self, value: float, setpoint: Optional[float] = None) -> float:
        if setpoint is not None:
            self.setpoint = setpoint
        error = value - self.setpoint
        P_out = self.kp * error
        self.integral += error
        I_out = self.ki * self.integral
        if self.error is not None:
            D_out = self.kd * (error - self.error)
        else:
            D_out = 0.0
        self.error = error
        self.last_value = P_out + I_out + D_out
        return self.last_value

    def clone(self) -> "PID":
        return PID(self.kp, self.ki, self.kd, self.setpoint, self.error, self.integral, self.last_value)
