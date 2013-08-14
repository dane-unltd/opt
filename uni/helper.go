package uni

import (
	"math"
	"time"
)

type Updater interface {
	Update(r *Result) Status
}

type helper struct {
	initialInterval float64
	initialTime     time.Time
	updates         []Updater
	x0, f0, d0      float64

	oldX    float64
	oldObjX float64
}

func newHelper(in *Solution) *helper {
	h := &helper{}
	h.initialTime = time.Now()
	h.initialInterval = in.UB - in.LB
	if math.IsInf(h.initialInterval, 1) {
		h.initialInterval = 0
	}

	h.x0 = in.LB
	h.f0 = in.ObjLB
	h.d0 = in.DerivLB

	h.oldX = math.NaN()
	h.oldObjX = math.NaN()
	return h
}

func (h *helper) update(r *Result, p *Params) Status {
	r.Time = time.Since(h.initialTime)
	if h.doUpdates(r); r.Status != 0 {
		return r.Status
	}
	if r.Status = h.checkConvergence(r, p); r.Status != 0 {
		return r.Status
	}

	h.oldX = r.X
	h.oldObjX = r.ObjX
	r.Iter++

	return r.Status
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

func (h *helper) checkConvergence(r *Result, p *Params) Status {
	if p.Inexact {
		if math.Abs(r.DerivX/h.d0) < p.Curvature &&
			r.ObjX-h.f0 < p.Armijo*(r.X-h.x0)*h.d0 {
			return WolfeConv
		}
	}
	if math.Abs(r.UB-r.LB) < p.XTolAbs {
		return XAbsConv
	}
	if math.Abs((r.UB-r.LB)/h.initialInterval) < p.XTolRel {
		return XRelConv
	}
	if math.Abs(r.DerivX) < p.FunTolAbs {
		return DerivAbsConv
	}
	if math.Abs(r.DerivX/h.d0) < p.FunTolRel {
		return DerivRelConv
	}
	if math.Abs(r.ObjX-h.oldObjX) < p.FunTolAbs {
		return ObjAbsConv
	}
	if math.Abs((r.ObjX-h.oldObjX)/r.ObjX) < p.FunTolRel {
		return ObjRelConv
	}

	if r.Iter > p.IterMax {
		return IterLimit
	}
	if r.Time > p.TimeMax {
		return TimeLimit
	}
	if r.FunEvals > p.FunEvalMax {
		return FunEvalLimit
	}
	return NotTerminated
}
