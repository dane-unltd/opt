package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"math"
	"time"
)

type Callback interface {
	Update(r *Result) Status
}

type helper struct {
	initialTime     time.Time
	initialGradNorm float64

	gradNorm float64

	oldX    mat.Vec
	oldObjX float64

	callbacks []Callback

	temp mat.Vec
}

func NewHelper(in *Solution) *helper {
	h := &helper{}
	h.initialTime = time.Now()

	if in.GradX != nil {
		h.initialGradNorm = in.GradX.Nrm2()
	} else {
		h.initialGradNorm = math.NaN()
	}

	h.oldX = mat.NewVec(len(in.X)).Scal(math.NaN())
	h.temp = mat.NewVec(len(in.X)).Scal(math.NaN())
	h.oldObjX = math.NaN()
	h.gradNorm = math.NaN()
	return h
}

func (h *helper) update(r *Result, p *Params) Status {
	r.Time = time.Since(h.initialTime)
	if r.GradX != nil {
		h.gradNorm = r.GradX.Nrm2()
	}
	if h.doCallbacks(r); r.Status != 0 {
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

func (h *helper) doCallbacks(r *Result) Status {
	for _, cb := range h.callbacks {
		st := cb.Update(r)
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
