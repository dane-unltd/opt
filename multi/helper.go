package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"math"
	"time"
)

type Updater interface {
	Update(r *Result) Status
}

type helper struct {
	initialTime     time.Time
	initialGradNorm float64

	gradNorm float64

	oldX    mat.Vec
	oldObjX float64

	updates []Updater

	temp mat.Vec
}

func newHelper(in *Solution, u []Updater) *helper {
	h := &helper{}
	h.initialTime = time.Now()
	h.updates = u

	if in.GradX != nil {
		h.initialGradNorm = in.GradX.Nrm2()
	} else {
		h.initialGradNorm = nan
	}

	h.oldX = mat.NewVec(len(in.X)).Scal(nan)
	h.temp = mat.NewVec(len(in.X)).Scal(nan)
	h.oldObjX = nan
	h.gradNorm = nan
	return h
}

func (h *helper) update(r *Result, p *Params) Status {
	r.Time = time.Since(h.initialTime)
	if r.GradX != nil {
		h.gradNorm = r.GradX.Nrm2()
	}
	if h.doUpdates(r); r.Status != 0 {
		return r.Status
	}
	if r.Status = h.checkConvergence(r, p); r.Status != 0 {
		return r.Status
	}

	h.oldX.Copy(r.X)
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
	if math.Abs(h.gradNorm) < p.FunTolAbs {
		return GradAbsConv
	}
	if math.Abs((h.gradNorm)/h.initialGradNorm) < p.FunTolRel {
		return GradRelConv
	}
	if math.Abs(r.ObjX-h.oldObjX) < p.FunTolAbs {
		return ObjAbsConv
	}
	if math.Abs((r.ObjX-h.oldObjX)/r.ObjX) < p.FunTolRel {
		return ObjRelConv
	}

	h.temp.Sub(r.X, h.oldX)
	if h.temp.Nrm2() < p.XTolAbs {
		return XAbsConv
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
