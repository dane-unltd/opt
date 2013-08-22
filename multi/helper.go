package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"math"
	"time"
)

type Updater interface {
	Update(r *Result) Status
}

type basicConv struct {
	initialTime     time.Time
	initialGradNorm float64

	gradNorm float64

	oldX    mat.Vec
	oldObjX float64

	temp   mat.Vec
	params *Params
}

func newBasicConv(in *Solution, p *Params) *basicConv {
	conv := &basicConv{}
	conv.initialTime = time.Now()
	conv.params = p

	if in.GradX != nil {
		conv.initialGradNorm = in.GradX.Nrm2()
	} else {
		conv.initialGradNorm = nan
	}

	conv.oldX = mat.NewVec(len(in.X)).Scal(nan)
	conv.temp = mat.NewVec(len(in.X)).Scal(nan)
	conv.oldObjX = nan
	conv.gradNorm = nan
	return conv
}

func (conv *basicConv) Update(r *Result) Status {
	r.Time = time.Since(conv.initialTime)
	if r.GradX != nil {
		conv.gradNorm = r.GradX.Nrm2()
	}
	if r.Status = conv.checkConvergence(r); r.Status != 0 {
		return r.Status
	}

	conv.oldX.Copy(r.X)
	conv.oldObjX = r.ObjX
	r.Iter++

	return r.Status
}

func (conv *basicConv) checkConvergence(r *Result) Status {
	if math.Abs(conv.gradNorm) < conv.params.FunTolAbs {
		return GradAbsConv
	}
	if math.Abs((conv.gradNorm)/conv.initialGradNorm) < conv.params.FunTolRel {
		return GradRelConv
	}
	if math.Abs(r.ObjX-conv.oldObjX) < conv.params.FunTolAbs {
		return ObjAbsConv
	}
	if math.Abs((r.ObjX-conv.oldObjX)/r.ObjX) < conv.params.FunTolRel {
		return ObjRelConv
	}

	conv.temp.Sub(r.X, conv.oldX)
	if conv.temp.Nrm2() < conv.params.XTolAbs {
		return XAbsConv
	}

	if r.Iter > conv.params.IterMax {
		return IterLimit
	}
	if r.Time > conv.params.TimeMax {
		return TimeLimit
	}
	if r.FunEvals > conv.params.FunEvalMax {
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
