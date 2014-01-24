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

	blasw.Dcopy(g, d)
	blasw.Dscal(-1, d)

	xTemp := blasw.NewVector(make([]float64, n))

	blasw.Dcopy(x, xTemp)
	blasw.Daxpy(s/2, d, xTemp)
	proj.Project(xTemp.Data)
	blasw.Daxpy(-1, x, xTemp)
	blasw.Dscal(2/s, xTemp)

	gLin := -blasw.Ddot(xTemp, xTemp)

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

		blasw.Daxpy(s, d, x)
		proj.Project(r.X)

		obj.DF(r.X, r.Grad)
		blasw.Dcopy(g, d)
		blasw.Dscal(-1, d)

		blasw.Dcopy(x, xTemp)
		blasw.Daxpy(s/2, d, xTemp)
		proj.Project(xTemp.Data)
		blasw.Daxpy(-1, x, xTemp)
		blasw.Dscal(2/s, xTemp)

		gLin = -blasw.Ddot(xTemp, xTemp)
	}
	return r
}
