package uni

import (
	"math"
	"time"
)

type Updater interface {
	Update(r *Result) Status
}

type basicConv struct {
	initialInterval float64
	initialTime     time.Time
	x0, f0, d0      float64

	oldX    float64
	oldObjX float64
}

func newBasicConv(in *Solution) *basicConv {
	conv := &basicConv{}
	conv.initialTime = time.Now()
	conv.initialInterval = in.UB - in.LB
	if math.IsInf(conv.initialInterval, 1) {
		conv.initialInterval = 0
	}

	conv.x0 = in.LB
	conv.f0 = in.ObjLB
	conv.d0 = in.DerivLB

	conv.oldX = nan
	conv.oldObjX = nan
	return conv
}

func (conv *basicConv) update(r *Result, p *Params) Status {
	if r.Status = conv.checkConvergence(r, p); r.Status != 0 {
		return r.Status
	}

	conv.oldX = r.X
	conv.oldObjX = r.ObjX
	r.Iter++

	return r.Status
}

func (conv *basicConv) checkConvergence(r *Result, p *Params) Status {
	if p.Inexact {
		if math.Abs(r.DerivX/conv.d0) < p.Curvature &&
			r.ObjX-conv.f0 < p.Armijo*(r.X-conv.x0)*conv.d0 {
			return WolfeConv
		}
	}
	if math.Abs(r.UB-r.LB) < p.XTolAbs {
		return XAbsConv
	}
	if math.Abs((r.UB-r.LB)/conv.initialInterval) < p.XTolRel {
		return XRelConv
	}
	if math.Abs(r.DerivX) < p.FunTolAbs {
		return DerivAbsConv
	}
	if math.Abs(r.DerivX/conv.d0) < p.FunTolRel {
		return DerivRelConv
	}
	if math.Abs(r.ObjX-conv.oldObjX) < p.FunTolAbs {
		return ObjAbsConv
	}
	if math.Abs((r.ObjX-conv.oldObjX)/r.ObjX) < p.FunTolRel {
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

func doUpdates(r *Result, upd []Updater) Status {
	for _, u := range upd {
		st := u.Update(r)
		if st != 0 {
			r.Status = st
		}
	}
	return r.Status
}
