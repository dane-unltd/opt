package multi

import (
	"github.com/dane-unltd/opt/uni"
	"github.com/gonum/blas/dbw"
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

	x := dbw.NewVector(r.X)
	g := dbw.NewVector(r.Grad)
	d := dbw.NewVector(make([]float64, n))

	dbw.Copy(g, d)
	dbw.Scal(-1, d)

	xTemp := dbw.NewVector(make([]float64, n))

	dbw.Copy(x, xTemp)
	dbw.Axpy(s/2, d, xTemp)
	proj.Project(xTemp.Data)
	dbw.Axpy(-1, x, xTemp)
	dbw.Scal(2/s, xTemp)

	gLin := -dbw.Dot(xTemp, xTemp)

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

		dbw.Axpy(s, d, x)
		proj.Project(r.X)

		obj.DF(r.X, r.Grad)
		dbw.Copy(g, d)
		dbw.Scal(-1, d)

		dbw.Copy(x, xTemp)
		dbw.Axpy(s/2, d, xTemp)
		proj.Project(xTemp.Data)
		dbw.Axpy(-1, x, xTemp)
		dbw.Scal(2/s, xTemp)

		gLin = -dbw.Dot(xTemp, xTemp)
	}
	return r
}
