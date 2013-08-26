package uni

import (
	"math"
	"time"
)

type Cubic struct {
	Termination
}

func NewCubic() *Cubic {
	return &Cubic{
		Termination: Termination{
			IterMax: 100,
			TimeMax: time.Minute,
		},
	}
}

func (sol *Cubic) Solve(o Deriv, in *Solution, upd ...Updater) *Result {
	r := NewResult(in)
	obj := ObjDerivWrapper{r: r, o: o}
	r.initDeriv(obj)

	initialTime := time.Now()

	upd = append(upd, sol.Termination)

	if r.DerivLower > 0 {
		r.Status = Fail
		return r
	}

	//search for upper bound
	for math.IsInf(r.XUpper, 1) {
		if r.Deriv > 0 {
			r.XUpper = r.X
			r.ObjUpper = r.Obj
			r.DerivUpper = r.Deriv
		} else {
			if r.Obj > r.ObjLower {
				r.XUpper = r.X
				r.ObjUpper = r.Obj
				r.DerivUpper = r.Deriv
			} else {
				lb := r.XLower
				r.XLower = r.X
				r.ObjLower = r.Obj
				r.DerivLower = r.Deriv

				r.X += 2 * (r.X - lb)
				r.Obj, r.Deriv = obj.ValDeriv(r.X)
				if doUpdates(r, initialTime, upd) != 0 {
					return r
				}
			}
		}
	}

	for {
		eps := 0.01 * (r.XUpper - r.XLower)

		//cubic interpolation between upper bound and lower bound
		eta := 3*(r.ObjLower-r.ObjUpper)/(r.XUpper-r.XLower) + r.DerivLower + r.DerivUpper
		nu := math.Sqrt(math.Pow(eta, 2) - r.DerivLower*r.DerivUpper)

		r.X = r.XLower + (r.XUpper-r.XLower)*
			(1-(r.DerivUpper+nu-eta)/(r.DerivUpper-r.DerivLower+2*nu))

		if !(r.X > r.XLower && r.X < r.XUpper) {
			r.X = (r.XUpper + r.XLower) / 2
		}
		if (r.X - r.XLower) < eps {
			r.X = r.XLower + eps
		}
		if (r.XUpper - r.X) < eps {
			r.X = r.XUpper - eps
		}

		r.Obj, r.Deriv = obj.ValDeriv(r.X)

		if r.Deriv > 0 {
			r.XUpper = r.X
			r.ObjUpper = r.Obj
			r.DerivUpper = r.Deriv
		} else {
			if r.Obj > r.ObjLower {
				r.XUpper = r.X
				r.ObjUpper = r.Obj
				r.DerivUpper = r.Deriv
			} else {
				r.XLower = r.X
				r.ObjLower = r.Obj
				r.DerivLower = r.Deriv
			}
		}

		if doUpdates(r, initialTime, upd) != 0 {
			break
		}
	}
	return r
}
