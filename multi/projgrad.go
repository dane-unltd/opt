package multi

import (
	"github.com/dane-unltd/opt/uni"
	"github.com/gonum/blas/dblas"
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

	x := dblas.NewVector(r.X)
	g := dblas.NewVector(r.Grad)
	d := dblas.NewVector(make([]float64, n))

	dblas.Dcopy(g, d)
	dblas.Dscal(-1, d)

	xTemp := dblas.NewVector(make([]float64, n))

	dblas.Dcopy(x, xTemp)
	dblas.Daxpy(s/2, d, xTemp)
	proj.Project(xTemp.Data)
	dblas.Daxpy(-1, x, xTemp)
	dblas.Dscal(2/s, xTemp)

	gLin := -dblas.Ddot(xTemp, xTemp)

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

		dblas.Daxpy(s, d, x)
		proj.Project(r.X)

		obj.DF(r.X, r.Grad)
		dblas.Dcopy(g, d)
		dblas.Dscal(-1, d)

		dblas.Dcopy(x, xTemp)
		dblas.Daxpy(s/2, d, xTemp)
		proj.Project(xTemp.Data)
		dblas.Daxpy(-1, x, xTemp)
		dblas.Dscal(2/s, xTemp)

		gLin = -dblas.Ddot(xTemp, xTemp)
	}
	return r
}
