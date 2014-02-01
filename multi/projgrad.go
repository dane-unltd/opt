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

	dblas.Copy(g, d)
	dblas.Scal(-1, d)

	xTemp := dblas.NewVector(make([]float64, n))

	dblas.Copy(x, xTemp)
	dblas.Axpy(s/2, d, xTemp)
	proj.Project(xTemp.Data)
	dblas.Axpy(-1, x, xTemp)
	dblas.Scal(2/s, xTemp)

	gLin := -dblas.Dot(xTemp, xTemp)

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

		dblas.Axpy(s, d, x)
		proj.Project(r.X)

		obj.DF(r.X, r.Grad)
		dblas.Copy(g, d)
		dblas.Scal(-1, d)

		dblas.Copy(x, xTemp)
		dblas.Axpy(s/2, d, xTemp)
		proj.Project(xTemp.Data)
		dblas.Axpy(-1, x, xTemp)
		dblas.Scal(2/s, xTemp)

		gLin = -dblas.Dot(xTemp, xTemp)
	}
	return r
}
