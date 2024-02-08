/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE468_Incorrect_Pointer_Scaling__int_19.c
Label Definition File: CWE468_Incorrect_Pointer_Scaling.label.xml
Template File: point-flaw-19.tmpl.c
*/
/*
 * @description
 * CWE: 468 Incorrect Pointer Scaling
 * Sinks: int
 *    GoodSink: Do not include sizeof(int) since pointer arithmetic is automatically scaled
 *    BadSink : Included sizeof(int) which is unnecessary since pointer arithmetic is automatically scaled
 * Flow Variant: 19 Control flow: Dead code after a return
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE468_Incorrect_Pointer_Scaling__int_19_bad()
{
    {
        int array[5] = { 1, 2, 3, 4, 5 };
        int *pointer = array;
        /* get array[2] */
        /* FLAW: included *sizeof(int) which is unnecessary since pointer arithmetic is automatically scaled */
        int toPrint = *(pointer+(2*sizeof(int)));
        printIntLine(toPrint);
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        int array[5] = { 1, 2, 3, 4, 5 };
        int *pointer = array;
        /* get array[2] */
        /* FIX: no sizeof() needed since pointer math is automatically scaled */
        int toPrint = *(pointer+2);
        printIntLine(toPrint);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() reverses the GoodSinkBody and the BadSinkBody so that the BadSinkBody never runs */
static void good1()
{
    {
        int array[5] = { 1, 2, 3, 4, 5 };
        int *pointer = array;
        /* get array[2] */
        /* FIX: no sizeof() needed since pointer math is automatically scaled */
        int toPrint = *(pointer+2);
        printIntLine(toPrint);
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        int array[5] = { 1, 2, 3, 4, 5 };
        int *pointer = array;
        /* get array[2] */
        /* FLAW: included *sizeof(int) which is unnecessary since pointer arithmetic is automatically scaled */
        int toPrint = *(pointer+(2*sizeof(int)));
        printIntLine(toPrint);
    }
}

void CWE468_Incorrect_Pointer_Scaling__int_19_good()
{
    good1();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE468_Incorrect_Pointer_Scaling__int_19_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE468_Incorrect_Pointer_Scaling__int_19_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
