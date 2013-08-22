package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
)

type ProjGrad struct {
	LineSearch uni.Solver
}

func NewProjGrad() *ProjGrad {
	s := &ProjGrad{
		LineSearch: uni.NewArmijo(),
	}
	return s
}

func (sol ProjGrad) Solve(o Grad, proj Projection, in *Solution, p *Params, upd ...Updater) *Result {
	r := NewResult(in)
	obj := ObjGradWrapper{r: r, o: o}
	r.initGrad(obj)
	upd = append(upd, newBasicConv(r.Solution, p))

	n := len(r.X)
	s := 1.0 //initial step size

	d := mat.NewVec(n)
	d.Copy(r.GradX)
	d.Scal(-1)

	xTemp := mat.NewVec(n)

	xTemp.Copy(r.X)
	xTemp.Axpy(s/2, d)
	proj.Project(xTemp)
	xTemp.Sub(xTemp, r.X)
	xTemp.Scal(2 / s)

	gLin := -xTemp.Nrm2Sq()

	lineFunc := NewLineFuncProj(obj, proj, r.X, d)
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
		proj.Project(r.X)

		obj.ValGrad(r.X, r.GradX)
		d.Copy(r.GradX)
		d.Scal(-1)

		xTemp.Copy(r.X)
		xTemp.Axpy(s/2, d)
		proj.Project(xTemp)
		xTemp.Sub(xTemp, r.X)
		xTemp.Scal(2 / s)

		gLin = -xTemp.Nrm2Sq()
	}
	return r
}
