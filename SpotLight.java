// SpotLight class

import javax.vecmath.*;

public class SpotLight extends Light
{
	SpotLight(Vector3f _from, Vector3f _to, float _exp, float _cutoff, Color3f intens)
	{
		from = new Vector3f(_from);
		to = new Vector3f(_to);
		dir = new Vector3f(from);
		dir.sub(to);
		dir.normalize();
		spot_cos = (float)Math.cos(_cutoff * Math.PI / 360.f);
		spot_exp = _exp;
		intensity = new Color3f(intens);
	}

	public Color3f getLight(Vector3f p, Vector3f lightPos, Vector3f lightDir)
	{
		lightPos.set(from);
		lightDir.set(lightPos);
		lightDir.sub(p);
		float r = lightDir.length();
		lightDir.normalize();
		float cosangle = lightDir.dot(dir);
		if (cosangle < spot_cos) {
			return null;
		}
		Color3f lightIntens = new Color3f(intensity);
		lightIntens.scale(1.f / (r*r));
		lightIntens.scale((float)Math.pow(cosangle, spot_exp));
		return lightIntens;
	}

	private Vector3f from;
	private Vector3f to;
	private float spot_cos;
	private float spot_exp;
	private Vector3f dir;
}


