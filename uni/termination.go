package uni

import (
	"time"
)

type IterMax int

func (it IterMax) Update(r *Result) Status {
	if r.Iter >= int(it) {
		return IterLimit
	}
	return NotTerminated
}

type TimeMax time.Duration

func (t TimeMax) Update(r *Result) Status {
	if r.Time >= time.Duration(t) {
		return TimeLimit
	}
	return NotTerminated
}
