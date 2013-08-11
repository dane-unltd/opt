package uni

import (
	"math"
)

type Cubic struct {
}

func NewCubic() *Cubic {
	return &Cubic{}
}

func (sol *Cubic) Solve(o Deriv, in *Solution, p *Params) *Result {
	r := NewResult(in)
	obj := ObjDerivWrapper{r: r, o: o}
	r.initDeriv(obj)
	h := NewHelper(r.Solution)

	eps := 0.4 * p.XTolAbs

	if r.DerivLB > 0 {
		r.Status = Fail
		return r
	}

	//search for upper bound
	for math.IsInf(r.UB, 1) {
		if r.DerivX > 0 {
			r.UB = r.X
			r.ObjUB = r.ObjX
			r.DerivUB = r.DerivX
		} else {
			if r.ObjX-h.f0 > p.Armijo*h.d0*(r.X-h.x0) {
				r.UB = r.X
				r.ObjUB = r.ObjX
				r.DerivUB = r.DerivX
			} else {
				lb := r.LB
				r.LB = r.X
				r.ObjLB = r.ObjX
				r.DerivLB = r.DerivX

				r.X += 2 * (r.X - lb)
				r.ObjX, r.DerivX = obj.ValDeriv(r.X)
				if h.update(r, p); r.Status != 0 {
					return r
				}
			}
		}
	}

	if eps == 0 {
		eps = p.XTolRel * (r.UB - r.LB)
	}
	if eps == 0 {
		eps = 1e-3 * (r.UB - r.LB)
	}

	for {
		//cubic interpolation between upper bound and lower bound
		eta := 3*(r.ObjLB-r.ObjUB)/(r.UB-r.LB) + r.DerivLB + r.DerivUB
		nu := math.Sqrt(math.Pow(eta, 2) - r.DerivLB*r.DerivUB)

		r.X = r.LB + (r.UB-r.LB)*
			(1-(r.DerivUB+nu-eta)/(r.DerivUB-r.DerivLB+2*nu))

		if !(r.X > r.LB && r.X < r.UB) {
			r.X = (r.UB + r.LB) / 2
		}
		if (r.X - r.LB) < eps {
			r.X = r.LB + eps
		}
		if (r.UB - r.X) < eps {
			r.X = r.UB - eps
		}

		r.ObjX, r.DerivX = obj.ValDeriv(r.X)

		if r.DerivX > 0 {
			r.UB = r.X
			r.ObjUB = r.ObjX
			r.DerivUB = r.DerivX
		} else {
			if r.ObjX-h.f0 > p.Armijo*h.d0*(r.X-h.x0) {
				r.UB = r.X
				r.ObjUB = r.ObjX
				r.DerivUB = r.DerivX
			} else {
				r.LB = r.X
				r.ObjLB = r.ObjX
				r.DerivLB = r.DerivX
			}
		}

		if h.update(r, p); r.Status != 0 {
			return r
		}
	}
	return r
}
