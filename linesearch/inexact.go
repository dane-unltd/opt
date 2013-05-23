package linesearch

//Inexact line search using Armijo's rule.
func Inexact(obj func(float64) float64, f0, g0, s float64) (float64, float64) {
	beta := 0.5

	f := obj(s)

	if f-f0 > 0.5*g0*s {
		fPrev := f
		s *= beta
		for {
			f = obj(s)
			if f-f0 <= 0.5*g0*s {
				if fPrev < f {
					s /= beta
					f = fPrev
				}
				break
			}
			fPrev = f
			s *= beta
		}
	} else {
		fPrev := f
		s /= beta
		for {
			f = obj(s)
			if f-f0 > 0.5*g0*s {
				if fPrev < f {
					s *= beta
					f = fPrev
				}
				break
			}
			fPrev = f
			s /= beta
		}
	}
	return s, f
}
