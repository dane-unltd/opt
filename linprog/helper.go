package linprog

import (
	"time"
)

type Updater interface {
	Update(r *Result) Status
}

type basicConv struct {
	initialTime time.Time
	params      *Params
}

func newBasicConv(p *Params) *basicConv {
	conv := &basicConv{}
	conv.initialTime = time.Now()
	conv.params = p
	return conv
}

func (conv *basicConv) Update(r *Result) Status {
	r.Time = time.Since(conv.initialTime)
	r.Iter++
	return conv.checkConvergence(r)
}

func (conv *basicConv) checkConvergence(r *Result) Status {
	if r.Iter > conv.params.IterMax {
		return IterLimit
	}
	if r.Time > conv.params.TimeMax {
		return TimeLimit
	}
	return NotTerminated
}

func doUpdates(r *Result, upd []Updater) Status {
	for _, u := range upd {
		st := u.Update(r)
		if st != 0 {
			r.Status = st
		}
	}
	return r.Status
}
