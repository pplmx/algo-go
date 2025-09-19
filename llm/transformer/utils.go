package transformer

import "time"

// 性能计时器
type Timer struct {
	start time.Time
	end   time.Time
}

func (t *Timer) Start() {
	t.start = time.Now()
}

func (t *Timer) Stop() {
	t.end = time.Now()
}

func (t *Timer) ElapsedMs() float64 {
	return float64(t.end.Sub(t.start).Nanoseconds()) / 1e6
}
