package uni

import (
	"math"
)

//Inexact line search using Armijo's rule.
type Armijo struct {
}

func NewArmijo() *Armijo {
	return &Armijo{}
}

func (s *Armijo) Solve(o Function, in *Solution, p *Params) *Result {
	r := NewResult(in)
	obj := ObjWrapper{r: r, o: o}
	r.init(obj)
	h := NewHelper(r.Solution)

	beta := 0.5

	step := r.X - r.LB
	maxStep := r.UB - r.LB

	if math.IsNaN(r.DerivLB) {
		panic("have to set derivation of lower bound for Armijo")
	}

	if r.DerivLB > 0 {
		r.Status = Fail
		return r
	}

	r.X = r.LB + step
	r.ObjX = obj.Val(r.X)

	if r.ObjX-r.ObjLB > p.Armijo*r.DerivLB*step {
		fPrev := r.ObjX
		step *= beta
		for {
			r.X = r.LB + step
			r.ObjX = obj.Val(r.X)

			if h.update(r, p); r.Status != 0 {
				break
			}

			if r.ObjX-r.ObjLB <= p.Armijo*r.DerivLB*step {
				if fPrev < r.ObjX {
					step /= beta
					r.X = r.LB + step
					r.ObjX = fPrev
				}
				break
			}
			fPrev = r.ObjX
			step *= beta
		}
	} else {
		fPrev := r.ObjX
		if step == maxStep {
			return r
		}
		step /= beta
		if step > maxStep {
			step = maxStep
		}
		for {
			r.X = r.LB + step
			r.ObjX = obj.Val(r.X)
			if h.update(r, p); r.Status != 0 {
				break
			}

			if r.ObjX-r.ObjLB > p.Armijo*r.DerivLB*step {
				if fPrev < r.ObjX {
					step *= beta
					r.X = r.LB + step
					r.ObjX = fPrev
				}
				break
			}
			fPrev = r.ObjX
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
