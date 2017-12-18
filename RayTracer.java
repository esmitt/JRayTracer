// RayTracer class

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.Scanner;
import java.util.Vector;

import javax.imageio.ImageIO;
import javax.vecmath.Color3f;
import javax.vecmath.Vector3f;

public class RayTracer {

	private Color3f image[][];	// image that stores floating point color
	private String image_name;	// output image name
	private int width, height;	// image width, height
	private int xsample, ysample;	// samples used for super sampling
	private Color3f background;	// background color
	private Color3f ambient;	// ambient color
	private int maxdepth;		// max recursion depth for recursive ray tracing
	private float exposure;		// camera exposure for the entire scene

	private Camera camera;
	private Vector<Material> materials = new Vector<Material> ();	// array of materials
	private Vector<Shape> shapes = new Vector<Shape> ();			// array of shapes
	private Vector<Light> lights = new Vector<Light> ();			// array of lights

	private final boolean USE_THREADS = true;
	private final int NUM_THREADS = 4;

	private void initialize() {
		width = 256;
		height = 256;
		xsample = 1;
		ysample = 1;
		maxdepth = 5;
		background = new Color3f(0,0,0);
		ambient = new Color3f(0,0,0);
		exposure = 1.0f;

		image_name = new String("output.png");

		camera = new Camera(new Vector3f(0,0,0), new Vector3f(0,-1,0), new Vector3f(0,1,0), 45.f, 1.f);

		// add a default material: diffuse material with constant 1 reflectance
		materials.add(Material.makeDiffuse(new Color3f(0,0,0), new Color3f(1,1,1)));
	}

	public static void main(String[] args) {
		if (args.length == 1) {
			new RayTracer(args[0]);
		} else {
			System.out.println("Usage: java RayTracer input.scene");
		}
	}

	private Color3f raytracing(Ray ray, int depth)
	{
		HitRecord closestHit = null;
		Shape closestShape = null;
		Color3f finalColor = new Color3f(0.0f, 0.0f, 0.0f);
		/* TODO: complete the ray tracing function
		 * Feel free to make new function as needed. For example, you may want to add a 'shading' function */

		if (depth >= maxdepth)
			return background;

		// For every shape in the scene.
		for (Shape shape : shapes) {
			// Calculate the intersection with this shape
			HitRecord hitRecord = shape.hit(ray, 0.001f, 10000.0f);

			// If there is an intersection, then check if it is nearer to the ray origin than the last
			// intersection found for this ray.
			if (hitRecord != null) {
				if (closestHit == null) {
					closestHit = hitRecord;
					closestShape = shape;
				} else if (hitRecord.t < closestHit.t) {
					closestHit = hitRecord;
					closestShape = shape;
				}
			}
		}

		// If the ray intersects something.
		if (closestHit != null) {
			assert closestShape != null;

			// Check if the intersection point is in shadow by tracing a shadow ray towards every light source.
			for (Light light : lights) {
				// Assume that the point is not in shadow.
				boolean inShadow = false;

				// Get the direction and position of the light source.
				Vector3f lightPos = new Vector3f();
				Vector3f lightDir = new Vector3f();
				light.getLight(closestHit.pos, lightPos, lightDir);

				// Create a shadow ray with origin at the intersection point.
				Ray shadowRay = new Ray(closestHit.pos, lightDir);

				// For every shape, check if the shadow ray intersects something
				for (Shape shape: shapes) {
					// No self-intersection.
					if (shape == closestShape)
						continue;

					// Verify if the point is lit or in shadow for this light source.
					Vector3f lightVector = new Vector3f(lightPos);
					lightVector.sub(closestHit.pos);
					inShadow = shape.shadowHit(shadowRay,  0.001f, lightVector.length());
					// If any shape obstruct this light source, then there is no point on 
					// searching for more intersections.
					if (inShadow)
						break;
				}

				// If the point is not in shadow, then evaluate the lightning model.
				if (!inShadow) {
					Color3f lightningColor = shade(closestHit, ray.d, light);
					finalColor.add(lightningColor);
				}

			} // for (Light light : lights)

			// Evaluate reflection if the material is reflective.
			if (closestHit.material.Kr.x > 0.0f || closestHit.material.Kr.x > 0.0f || closestHit.material.Kr.x > 0.0f) {
				// First calculate a reflected ray.
				Vector3f invDirection = new Vector3f();
				invDirection.negate(ray.d);
				Vector3f reflectionDir = reflect(invDirection, closestHit.normal);
				reflectionDir.normalize();
				Ray reflectedRay = new Ray(closestHit.pos, reflectionDir);
				// Ray trace recursively.
				Color3f reflectionColor = raytracing(reflectedRay, depth + 1);
				// Apply the material's reflective color.
				reflectionColor = colorProduct(closestHit.material.Kr, reflectionColor);
				finalColor.add(reflectionColor);
			}

			// Evaluate transmission if the material is transmissive.
			if (closestHit.material.Kt.x > 0.0f || closestHit.material.Kt.y > 0.0f || closestHit.material.Kt.z > 0.0f) {
				// First calculate the refracted ray.
				Vector3f transmissionDir = refract(ray.d, closestHit.normal, closestHit.material.ior);
				assert transmissionDir != null;
				transmissionDir.normalize();
				Ray refractedRay = new Ray(closestHit.pos, transmissionDir);
				// Ray trace recursively.
				Color3f transmisionColor = raytracing(refractedRay, depth + 1);
				// Apply the material's transmissive color. 
				transmisionColor = colorProduct(closestHit.material.Kt, transmisionColor);
				finalColor.add(transmisionColor);
			}

			// Return the calculated color for this pixel.
			return finalColor;
		} // if (closestHit != null)

		// If there is no intersection, then return the background color.
		return background;
	} // Color3f raytracing(Ray ray, int depth)

	/**
	 * Applies the Phong shading model.
	 * 
	 * @param hit the surface information for a ray intersection.
	 * @param incidenceDirection the direction of the ray that produced the intersection (MUST point TOWARDS the hit point).
	 * @param light the light source.
	 * @return the surface color after shading.
	 */
	private Color3f shade(HitRecord hit, Vector3f incidenceDirection, Light light) {
		Vector3f lightPos = new Vector3f();
		Vector3f lightDir = new Vector3f();
		Color3f lightIntensity = light.getLight(hit.pos, lightPos, lightDir);
		Vector3f invLightDir = new Vector3f();
		invLightDir.negate(lightDir);

		if (lightIntensity == null)
			return new Color3f(0.0f, 0.0f, 0.0f);

		// Diffuse component.
		Vector3f normal = new Vector3f(hit.normal);
		float nDotL = Math.max(normal.dot(lightDir), 0.0f);
		Color3f diffuseColor = new Color3f(lightIntensity);
		diffuseColor.scale(nDotL);

		// Specular component.
		Vector3f reflectedVector = reflect(invLightDir, hit.normal);
		float rDotL = (float) Math.pow(Math.max(reflectedVector.dot(incidenceDirection), 0.0f), hit.material.phong_exp);
		Color3f specularColor = new Color3f(lightIntensity);
		specularColor.scale(rDotL);

		// Ambient component
		Color3f ambientColor = new Color3f();
		ambientColor = colorProduct(ambient, hit.material.Ka);
		ambientColor.scale(1.0f / (float)lights.size());

		// Apply the surface material.
		Color3f diffuseMaterial = new Color3f(hit.material.Kd);
		diffuseMaterial.scale((float) (1.0 / Math.PI));
		diffuseColor = colorProduct(diffuseColor, diffuseMaterial);
		specularColor = colorProduct(specularColor, hit.material.Ks);

		// Aggregate the final color.
		Color3f shadedColor = new Color3f(0.0f, 0.0f, 0.0f);
		shadedColor.add(diffuseColor);
		shadedColor.add(specularColor);
		shadedColor.add(ambientColor);

		return shadedColor;
	} //Color3f shade(HitRecord hit, Vector3f incidenceDirection, Light light)

	/**
	 * Multiplies two colors component-wise.
	 * @param a the first color.
	 * @param b the second color.
	 * @return a {@link Color3f} C = (a.x * b.x, a.y * b.y, a.z * b.z)
	 */
	private Color3f colorProduct(Color3f a, Color3f b) {
		Color3f c = new Color3f();
		c.set(a.x * b.x, a.y * b.y, a.z * b.z);
		return c;
	}

	// reflect a direction (in) around a given normal
	/* NOTE: dir is assuming to point AWAY from the hit point
	 * if your ray direction is point INTO the hit point, you should flip
	 * the sign of the direction before calling reflect
	 */
	private Vector3f reflect(Vector3f dir, Vector3f normal)
	{
		Vector3f out = new Vector3f(normal);
		out.scale(2.f * dir.dot(normal));
		out.sub(dir);
		return out;
	}

	// refract a direction (in) around a given normal and 'index of refraction' (ior)
	/* NOTE: dir is assuming to point INTO the hit point
	 * (this is different from the reflect function above, which assumes dir is pointing away
	 */
	private Vector3f refract(Vector3f dir, Vector3f normal, float ior)
	{
		float mu;
		mu = (normal.dot(dir) < 0) ? 1.f / ior : ior;

		float cos_thetai = dir.dot(normal);
		float sin_thetai2 = 1.f - cos_thetai*cos_thetai;

		if (mu*mu*sin_thetai2 > 1.f) return null;
		float sin_thetar = mu*(float)Math.sqrt(sin_thetai2);
		float cos_thetar = (float)Math.sqrt(1.f - sin_thetar*sin_thetar);

		Vector3f out = new Vector3f(normal);
		if (cos_thetai > 0)
		{
			out.scale(-mu*cos_thetai+cos_thetar);
			out.scaleAdd(mu, dir, out);

		} else {

			out.scale(-mu*cos_thetai-cos_thetar);
			out.scaleAdd(mu, dir, out);
		}
		out.normalize();
		return out;
	}

	/**
	 * Creates a coordinate system from a normal vector.
	 * 
	 * This function returns two vectors that are orthogonal to the given normal vector.
	 * Algorithm from http://scratchapixel.com
	 * 
	 * @param normal the normal vector.
	 * @param nt the second basis of the coordinate system.
	 * @param nb the third basis of the coordinate system.
	 */
	void createCoordinateSystem(Vector3f normal, Vector3f nt, Vector3f nb) {
		if (Math.abs(normal.x) > Math.abs(normal.y))
			nt = new Vector3f(normal.z, 0.0f, -normal.x);
		else
			nt = new Vector3f(0.0f, -normal.z, normal.y);
		nt.normalize();
		nb = new Vector3f();
		nb.cross(normal, nt);
		nb.normalize();
	}

	public RayTracer(String scene_name) {
		// initialize and set default parameters
		initialize();

		// parse scene file
		parseScene(scene_name);

		// create floating point image
		image = new Color3f[width][height];

		if (USE_THREADS) {
			// If the number of rows in the final image is not exactly divisible by the number of threads, then
			// the final image may have missing rows.
			// This check is disabled if run without the "-ea" VM switch.
			assert height % NUM_THREADS == 0;

			LinkedList<Thread> threads = new LinkedList<Thread>();

			// Create NUM_THREADS threads with ID's starting from 0.
			for (int k = 0; k < NUM_THREADS; k++) {
				// Make the thread ID a final local variable.
				final int tid = k;

				// Create the thread.
				Thread thread = new Thread(
						// Create a Runnable for the thread.
						new Runnable() {
							@Override
							public void run() {
								int i, j;
								float x, y;
								// This thread will calculate height / NUM_THREADS rows of the image.
								for (j = (height / NUM_THREADS) * tid; j < (height / NUM_THREADS) * (tid + 1); j++)
								{
									y = (float)j / (float)height;
									System.out.print("\rray tracing... " + j*100/height + "%");
									for (i=0; i<width; i ++)
									{
										x = (float)i / (float)width;
										image[i][j] = raytracing(camera.getCameraRay(x, y), 0);
									}
								}
							}
						} // new Runnable()
						); // Thread thread = new Thread

				// Put the threads in a list and then start them.
				threads.add(thread);
				thread.start();
			} // for (int k = 0; k < NUM_THREADS; k++)

			// Wait for all threads to finish.
			for (Thread thread : threads)
				try {
					thread.join();
				} catch (InterruptedException e) { }

		} else { 
			// Process the scene normally without threads.
			int i, j;
			float x, y;
			for (j = 0; j < height; j++)
			{
				y = (float)j / (float)height;
				System.out.print("\rray tracing... " + j*100/height + "%");
				for (i=0; i<width; i ++)
				{
					x = (float)i / (float)width;
					image[i][j] = raytracing(camera.getCameraRay(x, y), 0);
				}
			}
		} // if (USE_THREADS)

		System.out.println("\rray tracing completed.                       ");

		writeImage();
	}

	private void parseScene(String scene_name)
	{
		File file = null;
		Scanner scanner = null;
		try {
			file = new File(scene_name);
			scanner = new Scanner(file);
		} catch (IOException e) {
			System.out.println("error reading from file " + scene_name);
			System.exit(0);
		}
		String keyword;
		while(scanner.hasNext()) {

			keyword = scanner.next();
			// skip the comment lines
			if (keyword.charAt(0) == '#') {
				scanner.nextLine();
				continue;
			}
			if (keyword.compareToIgnoreCase("image")==0) {

				image_name = scanner.next();
				width = scanner.nextInt();
				height = scanner.nextInt();
				exposure = Float.parseFloat(scanner.next());

			} else if (keyword.compareToIgnoreCase("camera")==0) {

				Vector3f eye = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
				Vector3f at  = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
				Vector3f up  = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
				float fovy = Float.parseFloat(scanner.next());
				float aspect_ratio = (float)width / (float)height;

				camera = new Camera(eye, at, up, fovy, aspect_ratio);

			} else if (keyword.compareToIgnoreCase("background")==0) {

				background = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));

			} else if (keyword.compareToIgnoreCase("ambient")==0) { 

				ambient = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));

			} else if (keyword.compareToIgnoreCase("maxdepth")==0) {

				maxdepth = scanner.nextInt();

			} else if (keyword.compareToIgnoreCase("light")==0) {

				// parse light
				parseLight(scanner);

			} else if (keyword.compareToIgnoreCase("material")==0) {

				// parse material
				parseMaterial(scanner);

			} else if (keyword.compareToIgnoreCase("shape")==0) {

				// parse shape
				parseShape(scanner);

			} else {
				System.out.println("undefined keyword: " + keyword);
			}
		}
		scanner.close();
	}

	private void parseLight(Scanner scanner)
	{
		String lighttype;
		lighttype = scanner.next();
		if (lighttype.compareToIgnoreCase("point")==0) {

			/* add a new point light */
			Vector3f pos = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Color3f intens = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			lights.add(new PointLight(pos, intens));

		} else if (lighttype.compareToIgnoreCase("spot")==0) {

			/* add a new spot light */
			Vector3f from = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Vector3f to = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			float spot_exponent = Float.parseFloat(scanner.next());
			float spot_cutoff = Float.parseFloat(scanner.next());
			Color3f intens = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));

			lights.add(new SpotLight(from, to, spot_exponent, spot_cutoff, intens));

		} else if (lighttype.compareToIgnoreCase("area")==0) {

			/* Add a new "area light" */
			Vector3f center = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			float area = Float.parseFloat(scanner.next());
			int xSamples = scanner.nextInt();
			int ySamples = scanner.nextInt();
			Color3f intens = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));

			// Scale the light intensity.
			intens.scale(1.0f / (float)(xSamples * ySamples));
			// Divide area in half.
			area /= 2.0f;

			// Generate light samples.
			for (int x = 0; x < xSamples; x++) {
				for (int y = 0; y < ySamples; y++) {
					// Generate two random numbers in [0, 1).
					float s = (float) Math.random();
					float t = (float) Math.random();

					// Interpolate between -(area / 2) and (area / 2).
					float sx = (-area * s) + (area * (1.0f - s));
					float sy = (-area * t) + (area * (1.0f - t));

					// Position the light source inside a rectangle of area and normal (0, 0, 1).
					Vector3f lightPos = new Vector3f(sx, sy, 0.0f);
					// Translate the position to the area light position.
					lightPos.add(center);

					// Add the light sample.
					lights.add(new PointLight(lightPos, intens));
				}
			}

		} else {
			System.out.println("undefined light type: " + lighttype);
		}
	}

	private void parseMaterial(Scanner scanner)
	{
		String mattype;
		mattype = scanner.next();
		if (mattype.compareToIgnoreCase("diffuse")==0) {

			Color3f ka = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Color3f kd = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			materials.add(Material.makeDiffuse(ka, kd));

		} else if (mattype.compareToIgnoreCase("specular")==0) {

			Color3f ka = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Color3f kd = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Color3f ks = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			float phong_exp = Float.parseFloat(scanner.next());
			materials.add(Material.makeSpecular(ka, kd, ks, phong_exp));

		} else if (mattype.compareToIgnoreCase("mirror")==0) {

			Color3f kr = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			materials.add(Material.makeMirror(kr));

		} else if (mattype.compareToIgnoreCase("glass")==0) {

			Color3f kr = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Color3f kt = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			float ior = Float.parseFloat(scanner.next());
			materials.add(Material.makeGlass(kr, kt, ior));

		} else if (mattype.compareToIgnoreCase("super")==0) {

			Color3f ka = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Color3f kd = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Color3f ks = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			float phong_exp = Float.parseFloat(scanner.next());
			Color3f kr = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Color3f kt = new Color3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			float ior = Float.parseFloat(scanner.next());
			materials.add(Material.makeSuper(ka, kd, ks, phong_exp, kr, kt, ior));			
		}

		else {
			System.out.println("undefined material type: " + mattype);
		}

	}

	private void parseShape(Scanner scanner)
	{
		String shapetype;
		shapetype = scanner.next();
		Material material = materials.lastElement();
		if (shapetype.compareToIgnoreCase("plane")==0) {

			Vector3f P0 = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Vector3f N = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			shapes.add(new Plane(P0, N, material));

		} else if (shapetype.compareToIgnoreCase("sphere")==0) {

			Vector3f center = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			float radius = Float.parseFloat(scanner.next());
			shapes.add(new Sphere(center, radius, material));

		} else if (shapetype.compareToIgnoreCase("triangle")==0) {

			Vector3f p0 = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Vector3f p1 = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Vector3f p2 = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			shapes.add(new Triangle(p0, p1, p2, material));

		} else if (shapetype.compareToIgnoreCase("triangle_n")==0) {

			Vector3f p0 = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Vector3f p1 = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Vector3f p2 = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));

			Vector3f n0 = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Vector3f n1 = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));
			Vector3f n2 = new Vector3f(Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()), Float.parseFloat(scanner.next()));

			shapes.add(new Triangle(p0, p1, p2, n0, n1, n2, material));

		} else if (shapetype.compareToIgnoreCase("trimesh")==0) {

			TriMesh	mesh = new TriMesh();
			mesh.load(scanner.next());

			if (mesh.type.compareToIgnoreCase("triangle")==0) {
				int i;
				int idx0, idx1, idx2;
				for (i=0; i<mesh.faces.length/3; i++) {
					idx0 = mesh.faces[i*3+0];
					idx1 = mesh.faces[i*3+1];
					idx2 = mesh.faces[i*3+2];
					shapes.add(new Triangle(mesh.verts[idx0], mesh.verts[idx1], mesh.verts[idx2], material));
				}

			} else if (mesh.type.compareToIgnoreCase("triangle_n")==0) {
				int i;
				int idx0, idx1, idx2;
				for (i=0; i<mesh.faces.length/3; i++) {
					idx0 = mesh.faces[i*3+0];
					idx1 = mesh.faces[i*3+1];
					idx2 = mesh.faces[i*3+2];
					shapes.add(new Triangle(mesh.verts[idx0], mesh.verts[idx1], mesh.verts[idx2],
							mesh.normals[idx0], mesh.normals[idx1], mesh.normals[idx2],
							material));
				}

			} else {
				System.out.println("undefined trimesh type: " + mesh.type);
			}


		} else {
			System.out.println("undefined shape type: " + shapetype);
		}
	}

	// write image to a disk file
	// image will be multiplied by exposure
	private void writeImage() {
		int x, y, index;
		int pixels[] = new int[width * height];

		index = 0;
		// apply a standard 2.2 gamma correction
		float gamma = 1.f / 2.2f;
		for (y=height-1; y >= 0; y --) {
			for (x=0; x<width; x ++) {
				Color3f c = new Color3f(image[x][y]);
				c.x = (float)Math.pow(c.x*exposure, gamma);
				c.y = (float)Math.pow(c.y*exposure, gamma);
				c.z = (float)Math.pow(c.z*exposure, gamma);
				c.clampMax(1.f);
				pixels[index++] = c.get().getRGB();

			}
		}

		BufferedImage oimage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		oimage.setRGB(0, 0, width, height, pixels, 0, width);
		File outfile = new File(image_name);
		try {
			ImageIO.write(oimage, "png", outfile);
		} catch(IOException e) {
		}
	}
}
