package ch.zhaw.hassebjo;

import java.util.Arrays;

public class Test {

	public static void main(String[] args) {

		String str = "24	 Of course.	NULL";
//		String[] ar = str.split("\\P{L}+");
		String[] ar = str.split("\t");
		System.out.println(Arrays.asList(ar));
		
//		System.out.println("¦" + ar[1] + "¦");

	}

}
