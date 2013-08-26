package multi

import (
	"time"
)

type Termination struct {
	IterMax int
	TimeMax time.Duration
}

func (t Termination) Update(r *Result) Status {
	if r.Iter >= t.IterMax {
		return IterLimit
	}
	if r.Time >= t.TimeMax {
		return TimeLimit
	}
	return NotTerminated
}
