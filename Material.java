// Material class
// defines material which encapsulates Diffuse, Specular,
// Reflective and Refractive parameters

import javax.vecmath.*;

public class Material {

	public Color3f Ka;	// ambient reflectance
	public Color3f Kd;	// diffuse reflectance
	public Color3f Ks;	// specular reflectance
	public Color3f Kr;	// reflective color
	public Color3f Kt;	// transmitive (refractive) color
	public float phong_exp;		// phong specular exponent
	public float ior;			// index of refraction

	public Material()
	{
		Ka = new Color3f(0,0,0);
		Kd = new Color3f(0,0,0);
		Ks = new Color3f(0,0,0);
		Kr = new Color3f(0,0,0);
		Kt = new Color3f(0,0,0);
		phong_exp = 1.f;
		ior = 1.f;
	}
	static public Material makeDiffuse(Color3f a, Color3f d) {
		Material m = new Material();
		m.Ka = new Color3f(a);
		m.Kd = new Color3f(d);
		return m;
	}
	static public Material makeSpecular(Color3f a, Color3f d, Color3f s, float _exp) {
		Material m = new Material();
		m.Ka = new Color3f(a);
		m.Kd = new Color3f(d);
		m.Ks = new Color3f(s);
		m.phong_exp = _exp;
		return m;
	}
	static public Material makeMirror(Color3f r) {
		Material m = new Material();
		m.Kr = new Color3f(r);
		return m;
	}
	static public Material makeGlass(Color3f r, Color3f t, float _ior) {
		Material m = new Material();
		m.Kr = new Color3f(r);
		m.Kt = new Color3f(t);
		m.ior = _ior;
		return m;
	}
	static public Material makeSuper(Color3f a, Color3f d, Color3f s, float _exp, Color3f r, Color3f t, float _ior) {
		Material m = new Material();
		m.Ka = new Color3f(a);
		m.Kd = new Color3f(d);
		m.Ks = new Color3f(s);
		m.phong_exp = _exp;
		m.Kr = new Color3f(r);
		m.Kt = new Color3f(t);
		m.ior = _ior;
		return m;
	}
}

