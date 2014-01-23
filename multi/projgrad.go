package multi

import (
	"github.com/dane-unltd/opt/uni"
	"github.com/gonum/blas/blasw"
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

	x := blasw.NewVector(r.X)
	g := blasw.NewVector(r.Grad)
	d := blasw.NewVector(make([]float64, n))

	blasw.Copy(g, d)
	blasw.Scal(-1, d)

	xTemp := blasw.NewVector(make([]float64, n))

	blasw.Copy(x, xTemp)
	blasw.Axpy(s/2, d, xTemp)
	proj.Project(xTemp.Data)
	blasw.Axpy(-1, x, xTemp)
	blasw.Scal(2/s, xTemp)

	gLin := -blasw.Dot(xTemp, xTemp)

	lineFunc := NewLineFProj(obj, proj, r.X, d.Data)
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

		blasw.Axpy(s, d, x)
		proj.Project(r.X)

		obj.DF(r.X, r.Grad)
		blasw.Copy(g, d)
		blasw.Scal(-1, d)

		blasw.Copy(x, xTemp)
		blasw.Axpy(s/2, d, xTemp)
		proj.Project(xTemp.Data)
		blasw.Axpy(-1, x, xTemp)
		blasw.Scal(2/s, xTemp)

		gLin = -blasw.Dot(xTemp, xTemp)
	}
	return r
}
