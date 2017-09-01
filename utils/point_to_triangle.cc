#include "point_to_triangle.h"


double PointToTriangle(MPoint& pp, MPoint* tri) {
	MVector P = MVector(pp);
	MVector B = MVector(tri[0]);
	MVector E0 = MVector(tri[1]);
	MVector E1 = MVector(tri[2]);

	MVector D = B - P;
	double a = E0 * E0;
	double b = E0 * E1;
	double c = E1 * E1;
	double d = E0 * D;
	double e = E1 * D;
	double f = D * D;

	double det = a * c - b * b;
	double s = b * e - c * d;
	double t = b * d - a * e;
	double sqr_dist, inv_det, tmp0, tmp1, numer, denom;

	if (s + t <= det) {
		if (s < 0) {
			if (t < 0) {
				//region 4
				if (d < 0) {
					t = 0;
					if (-d >= a) {
						s = 1;
						sqr_dist = a + 2 * d + f;
					} else {
						s = -d / a;
						sqr_dist = d * s + f;
					}
				} else {
					s = 0;
					if (e >= 0) {
						t = 0;
						sqr_dist = f;
					} else {
						if (-e >= c) {
							t = 1;
							sqr_dist = c + 2 * e + f;
						} else {
							t = -e / c;
							sqr_dist = e * t + f;
						}
					}
				}
			} else {
				// region 3
				s = 0;
				if (e >= 0) {
					t = 0;
					sqr_dist = f;
				} else {
					if (-e >= c) {
						t = 1;
						sqr_dist = c + 2 * e + f;
					} else {
						t = -e / c;
						sqr_dist = e * t + f;
					}
				}
			}
		} else {
			if (t < 0) {
				// region 5
				t = 0;
				if (d >= 0) {
					s = 0;
					sqr_dist = f;
				} else {
					if (-d >= a) {
						s = 1;
						sqr_dist = a + 2 * d + f;
					} else {
						s = -d / a;
						sqr_dist = d * s + f;
					}
				}
			} else {
				// region 0
				inv_det = 1 / det;
				s = s * inv_det;
				t = t * inv_det;
				sqr_dist = (s * (a * s + b * t + 2 * d) +
					t * (b * s + c * t + 2 * e) + f);
			}
		}
	} else {
		if (s < 0) {
			// region 2
			tmp0 = b + d;
			tmp1 = c + e;
			if (tmp1 > tmp0) {
				numer = tmp1 - tmp0;
				denom = a - 2 * b + c;
				if (numer >= denom) {
					s = 1;
					t = 0;
					sqr_dist = a + 2 * d + f;
				} else {
					s = numer / denom;
					t = 1-s;
					sqr_dist = (s * (a * s + b * t + 2 * d) +
						t * (b * s + c * t + 2 * e) + f);
				}
			} else {
				s = 0;
				if (tmp1 <= 0) {
					t = 1;
					sqr_dist = c + 2 * e + f;
				} else {
					if (e >= 0) {
						t = 0;
						sqr_dist = f;
					} else {
						t = -e / c;
						sqr_dist = e * t + f;
					}
				}
			}
		} else {
			if (t < 0) {
				// region 6
				tmp0 = b + e;
				tmp1 = a + d;
				if (tmp1 > tmp0) {
					numer = tmp1 - tmp0;
					denom = a - 2 * b + c;
					if (numer >= denom) {
						s = 1;
						t = 0;
						sqr_dist = a + 2 * e + f;
					} else {
						s = numer / denom;
						t = 1 - t;
						sqr_dist = (s * (a * s + b * t + 2 * d) +
							t * (b * s + c * t + 2 * e) + f);
					}
				} else {
					t = 0;
					if (tmp1 <= 0) {
						s = 1;
						sqr_dist = a + 2 * d + f;
					} else {
						if (d >= 0) {
							s = 0;
							sqr_dist = f;
						} else {
							s = -d / a;
							sqr_dist = d * s + f;
						}
					}
				}
			} else {
				// region 1
				numer = c + e - b - d;
				if (numer <= 0) {
					s = 0;
					t = 1;
					sqr_dist = c + 2 * e + f;
				} else {
					denom = a - 2 * b + c;
					if (numer >= denom) {
						s = 1;
						t = 0;
						sqr_dist = a + 2 * d + f;
					} else {
						s = numer/denom;
						t = 1-s;
						sqr_dist = (s * (a * s + b * t + 2 * d) +
							t * (b * s + c * t + 2 * e) + f);
					}
				}
			}
		}
	}

	if (sqr_dist < 0) sqr_dist = 0;
	return sqrt(sqr_dist);
}

