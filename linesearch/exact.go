package linesearch

//Exact line search for strictly quasi-convex functions
func Exact(obj func(float64) float64, f0, sNew, zeta float64) (float64, float64) {
	s1 := 0.0
	f1 := f0

	f2, f3 := 0.0, 0.0
	s2, s3 := 0.0, 0.0

	fNew := obj(sNew)

	if fNew < f1 {
		f2 = fNew
		s2 = sNew

		s3 = 2 * s2
		f3 = obj(s3)
		for f3 <= f2 {
			s3 *= 2
			f3 = obj(s3)

		}
	} else {
		f3 = fNew
		s3 = sNew

		s2 = 0.5 * s3
		f2 = obj(s2)
		for f2 >= f1 {
			s2 *= 0.5
			f2 = obj(s2)
		}
	}

	for s3-s1 > zeta {
		sNew = -0.5 * (s3*s3*(f1-f2) + s2*s2*(f3-f1) + s1*s1*(f2-f3)) /
			(s3*(f2-f1) + s2*(f1-f3) + s1*(f3-f2))
		fNew = obj(sNew)
		if sNew > s2 {
			if fNew >= f2 {
				s3 = sNew
			} else {
				s1 = s2
				s2 = sNew
				f2 = fNew
			}
		} else {
			if fNew >= f2 {
				s1 = sNew
			} else {
				s3 = s2
				s2 = sNew
				f2 = fNew
			}
		}
	}
	return sNew, fNew
}
