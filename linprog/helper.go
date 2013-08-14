package linprog

import (
	"time"
)

type Updater interface {
	Update(r *Result) Status
}

type helper struct {
	initialTime time.Time
	updates     []Updater
}

func newHelper(u []Updater) *helper {
	h := &helper{}
	h.initialTime = time.Now()
	h.updates = u
	return h
}

func (h *helper) doUpdates(r *Result) Status {
	for _, u := range h.updates {
		st := u.Update(r)
		if st != 0 {
			r.Status = st
		}
	}
	return r.Status
}

func (h *helper) update(r *Result, p *Params) Status {
	r.Time = time.Since(h.initialTime)
	if h.doUpdates(r); r.Status != 0 {
		return r.Status
	}
	if r.Status = h.checkConvergence(r, p); r.Status != 0 {
		return r.Status
	}
	r.Iter++

	return r.Status
}

func (h *helper) checkConvergence(r *Result, p *Params) Status {
	if r.Iter > p.IterMax {
		return IterLimit
	}
	if r.Time > p.TimeMax {
		return TimeLimit
	}
	return NotTerminated
}
