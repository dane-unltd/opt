package uni

import (
	"math"
	"time"
)

//Inexact line search using Armijo's rule.
type Armijo struct {
	Armijo float64
	Termination
}

func NewArmijo() *Armijo {
	return &Armijo{
		Armijo: 0.2,
		Termination: Termination{
			IterMax: 100,
			TimeMax: time.Minute,
		},
	}
}

func (s *Armijo) Solve(o Function, in *Solution, upd ...Updater) *Result {
	r := NewResult(in)
	obj := ObjWrapper{r: r, o: o}
	r.init(obj)

	upd = append(upd, s.Termination)

	initialTime := time.Now()

	beta := 0.5

	step := r.X - r.XLower
	maxStep := r.XUpper - r.XLower

	if math.IsNaN(r.DerivLower) {
		panic("have to set derivation of lower bound for Armijo")
	}

	if r.DerivLower > 0 {
		r.Status = Fail
		return r
	}

	r.X = r.XLower + step
	r.Obj = obj.Val(r.X)

	if r.Obj-r.ObjLower > s.Armijo*r.DerivLower*step {
		fPrev := r.Obj
		step *= beta
		for {
			r.X = r.XLower + step
			r.Obj = obj.Val(r.X)

			if doUpdates(r, initialTime, upd) != 0 {
				break
			}

			if r.Obj-r.ObjLower <= s.Armijo*r.DerivLower*step {
				if fPrev < r.Obj {
					step /= beta
					r.X = r.XLower + step
					r.Obj = fPrev
				}
				break
			}
			fPrev = r.Obj
			step *= beta
		}
	} else {
		fPrev := r.Obj
		if step == maxStep {
			return r
		}
		step /= beta
		if step > maxStep {
			step = maxStep
		}
		for {
			r.X = r.XLower + step
			r.Obj = obj.Val(r.X)
			if doUpdates(r, initialTime, upd) != 0 {
				break
			}

			if r.Obj-r.ObjLower > s.Armijo*r.DerivLower*step {
				if fPrev < r.Obj {
					step *= beta
					r.X = r.XLower + step
					r.Obj = fPrev
				}
				break
			}
			fPrev = r.Obj
			if step == maxStep {
				return r
			}
			step /= beta
			if step > maxStep {
				step = maxStep
			}
		}
	}
	r.Status = Success
	return r
}
