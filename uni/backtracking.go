package uni

import (
	"math"
	"time"
)

//Inexact line search using Armijo's rule.
type Backtracking struct {
	Termination
	Armijo           float64
	StepMin, StepMax float64
}

func NewBacktracking() *Backtracking {
	return &Backtracking{
		Termination: Termination{
			IterMax: 1000,
			TimeMax: time.Minute,
		},
		Armijo:  0.2,
		StepMin: 0.2,
		StepMax: 0.8,
	}
}

func (sol *Backtracking) OptimizeFdF(o FdF, in *Solution, upd ...Updater) *Result {
	if math.IsNaN(in.DerivLower) {
		newIn := *in
		newIn.DerivLower = o.DF(newIn.XLower)
		return sol.OptimizeF(o, &newIn, upd...)
	}
	return sol.OptimizeF(o, in, upd...)
}

func (sol *Backtracking) OptimizeF(o F, in *Solution, upd ...Updater) *Result {
	r := NewResult(in)
	obj := fWrapper{r: r, f: o}
	r.initF(obj)

	upd = append(upd, sol.Termination)

	initialTime := time.Now()

	step := r.X - r.XLower

	if math.IsNaN(r.DerivLower) {
		panic("have to set derivation of lower bound for Armijo")
	}

	if r.DerivLower > 0 {
		r.Status = Fail
		return r
	}

	r.X = r.XLower + step
	r.Obj = obj.F(r.X)

	if r.Obj-r.ObjLower > sol.Armijo*r.DerivLower*step {
		//quadratic interpolation
		stepNew := -0.5 * r.DerivLower * step * step / (r.Obj - r.ObjLower - r.DerivLower*step)
		if !(stepNew/step > sol.StepMin) {
			step = sol.StepMin * step
		} else if !(stepNew/step < sol.StepMax) {
			step = sol.StepMax * step
		}

		r.XUpper = r.X
		r.ObjUpper = r.Obj

		r.X = r.XLower + step
		r.Obj = obj.F(r.X)

		if doUpdates(r, initialTime, upd) != 0 {
			return r
		}

		//Continue with cubic interpolation
		for r.Obj-r.ObjLower > sol.Armijo*r.DerivLower*step {
			step0 := step
			step1 := r.XUpper - r.XLower
			step0Sq := step0 * step0
			step1Sq := step1 * step1

			frac := 1 / (step0Sq * step1Sq * (step1 - step0))
			d1 := r.Obj - r.ObjLower - r.DerivLower*step1
			d0 := r.ObjUpper - r.ObjLower - r.DerivLower*step0

			a := frac * (step0Sq*d1 - step1Sq*d0)
			b := frac * (step1Sq*step1*d0 - step0Sq*step0*d1)

			stepNew := (-b + math.Sqrt(b*b-3*a*r.DerivLower)) / (3 * a)
			if !(stepNew/step > sol.StepMin) {
				step = sol.StepMin * step
			} else if !(stepNew/step < sol.StepMax) {
				step = sol.StepMax * step
			}

			r.XUpper = r.X
			r.ObjUpper = r.Obj

			r.X = r.XLower + step
			r.Obj = obj.F(r.X)

			if doUpdates(r, initialTime, upd) != 0 {
				return r
			}
		}
	}
	r.Status = Success
	return r
}
