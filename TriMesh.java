// TriMesh class
// defines a triangular mesh

import javax.vecmath.*;
import java.io.*;
import java.util.*;

public class TriMesh {

	public Vector3f[] verts = null;
	public Vector3f[] normals = null;
	public int[] faces = null;
	public String type;

	public void load(String filename) {
		File file = null;
		Scanner scanner = null;
		try {
			file = new File(filename);
			scanner = new Scanner(file);
		} catch (IOException e) {
			System.out.println("error reading from file " + filename);
			System.exit(0);
		}
		type = scanner.next();
		if (scanner.next().compareToIgnoreCase("vertex")!=0) {
			System.out.println("file " + filename + " has incorrect format");
			System.exit(0);
		}
		int nverts = scanner.nextInt();
		if (scanner.next().compareToIgnoreCase("face")!=0) {
			System.out.println("file " + filename + " has incorrect format");
			System.exit(0);
		}
		int nfaces = scanner.nextInt();
		verts = new Vector3f[nverts];
		faces = new int[nfaces*3];
		if (type.compareToIgnoreCase("triangle_n")==0) {
			normals = new Vector3f[nverts];
		}

		int i;
		for (i=0; i<nverts; i++) {
			verts[i] = new Vector3f(scanner.nextFloat(), scanner.nextFloat(), scanner.nextFloat());
			if (type.compareToIgnoreCase("triangle_n")==0) {
				normals[i] = new Vector3f(scanner.nextFloat(), scanner.nextFloat(), scanner.nextFloat());
			}
		}
		int idx0, idx1, idx2;
		for (i=0; i<nfaces; i++) {
			scanner.nextInt();
			idx0 = scanner.nextInt();
			idx1 = scanner.nextInt();
			idx2 = scanner.nextInt();
			faces[i*3+0] = idx0;
			faces[i*3+1] = idx1;
			faces[i*3+2] = idx2;
		
		}
		scanner.close();
	}

}
