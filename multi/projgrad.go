package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
	"time"
)

type ProjGrad struct {
	Termination
	LineSearch *uni.Backtracking
}

func NewProjGrad() *ProjGrad {
	s := &ProjGrad{
		Termination: Termination{
			IterMax: 10000,
			TimeMax: time.Minute,
		},
		LineSearch: uni.NewBacktracking(),
	}
	return s
}

func (sol ProjGrad) OptimizeFProj(o FdF, proj Projection, in *Solution, upd ...Updater) *Result {
	r := NewResult(in)
	obj := fdfWrapper{r: r, fdf: o}
	r.initFdF(obj)

	upd = append(upd, sol.Termination)

	initialTime := time.Now()

	n := len(r.X)
	s := 1.0 //initial step size

	d := mat.NewVec(n)
	d.Copy(r.Grad)
	d.Scal(-1)

	xTemp := mat.NewVec(n)

	xTemp.Copy(r.X)
	xTemp.Axpy(s/2, d)
	proj.Project(xTemp)
	xTemp.Sub(xTemp, r.X)
	xTemp.Scal(2 / s)

	gLin := -xTemp.Nrm2Sq()

	lineFunc := NewLineFProj(obj, proj, r.X, d)
	lsInit := uni.NewSolution()

	for doUpdates(r, initialTime, upd) == 0 {
		lsInit.Set(s)
		lsInit.SetLower(0, r.Obj, gLin)
		lsRes := sol.LineSearch.OptimizeF(lineFunc, lsInit)
		if lsRes.Status < 0 {
			r.Status = Status(lsRes.Status)
			break
		}
		s, r.Obj = lsRes.X, lsRes.Obj

		r.X.Axpy(s, d)
		proj.Project(r.X)

		obj.DF(r.X, r.Grad)
		d.Copy(r.Grad)
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
