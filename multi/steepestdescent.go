package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
)

type SteepestDescent struct {
	LineSearch uni.DerivSolver
}

func NewSteepestDescent() *SteepestDescent {
	s := &SteepestDescent{
		LineSearch: uni.DerivWrapper{uni.NewArmijo()},
	}
	return s
}

func (sol SteepestDescent) Solve(o Grad, in *Solution, p *Params, upd ...Updater) *Result {
	r := NewResult(in)
	obj := ObjGradWrapper{r: r, o: o}
	r.initGrad(obj)
	upd = append(upd, newBasicConv(r.Solution, p))

	n := len(r.X)
	s := 1.0 //initial step size
	gLin := -r.GradX.Nrm2Sq()

	d := mat.NewVec(n)
	d.Copy(r.GradX)
	d.Scal(-1)

	lineFunc := NewLineFuncDeriv(obj, r.X, d)
	lsInit := uni.NewSolution()
	lsParams := uni.NewParams()

	for doUpdates(r, upd) == 0 {
		lsInit.SetX(s)
		lsInit.SetLB(0, r.ObjX, gLin)
		lsRes := sol.LineSearch.Solve(lineFunc, lsInit, lsParams)
		if lsRes.Status < 0 {
			r.Status = Status(lsRes.Status)
			break
		}
		s, r.ObjX = lsRes.X, lsRes.ObjX

		r.X.Axpy(s, d)

		obj.ValGrad(r.X, r.GradX)
		d.Copy(r.GradX)
		d.Scal(-1)

		gLin = -d.Nrm2Sq()
	}
	return r
}
