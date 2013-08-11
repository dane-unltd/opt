package linprog

import (
	"time"
)

type Callback interface {
	Update(r *Result) Status
}

type helper struct {
	initialTime time.Time
	callbacks   []Callback
}

func NewHelper(cb []Callback) *helper {
	h := &helper{}
	h.initialTime = time.Now()
	h.callbacks = cb
	return h
}

func (h *helper) doCallbacks(r *Result) Status {
	for _, cb := range h.callbacks {
		st := cb.Update(r)
		if st != 0 {
			r.Status = st
		}
	}
	return r.Status
}

func (h *helper) update(r *Result, p *Params) Status {
	r.Time = time.Since(h.initialTime)
	if h.doCallbacks(r); r.Status != 0 {
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
