/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE191_Integer_Underflow__int_fscanf_19.c
Label Definition File: CWE191_Integer_Underflow__int.label.xml
Template File: sources-sinks-19.tmpl.c
*/
/*
 * @description
 * CWE: 191 Integer Underflow
 * BadSource: fscanf Read data from the console using fscanf()
 * GoodSource: Greater than INT_MIN
 * Sinks:
 *    GoodSink: Ensure there is no underflow before performing the subtraction
 *    BadSink : Subtract 1 from data
 * Flow Variant: 19 Control flow: Dead code after a return
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE191_Integer_Underflow__int_fscanf_19_bad()
{
    int data;
    /* Initialize data */
    data = -1;
    fscanf (stdin, "%d", &data);
    {
        /* POTENTIAL FLAW: Subtracting 1 from data could cause an integer underflow */
        int result = data - 1;
        printIntLine(result);
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        int result = 0;
        /* FIX: Add a check to prevent an underflow from occurring */
        if (data > INT_MIN)
        {
            result = data - 1;
            printIntLine(result);
        }
        else
        {
            printLine("Input value is too small to perform arithmetic safely.");
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G() - use badsource and goodsink by switching statements around return */
static void goodB2G()
{
    int data;
    /* Initialize data */
    data = -1;
    fscanf (stdin, "%d", &data);
    {
        int result = 0;
        /* FIX: Add a check to prevent an underflow from occurring */
        if (data > INT_MIN)
        {
            result = data - 1;
            printIntLine(result);
        }
        else
        {
            printLine("Input value is too small to perform arithmetic safely.");
        }
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        /* POTENTIAL FLAW: Subtracting 1 from data could cause an integer underflow */
        int result = data - 1;
        printIntLine(result);
    }
}

/* goodG2B() - use goodsource and badsink by moving BadSource and BadSink to after return */
static void goodG2B()
{
    int data;
    /* Initialize data */
    data = -1;
    /* FIX: Use a small value greater than the min value for this data type */
    data = 5;
    {
        /* POTENTIAL FLAW: Subtracting 1 from data could cause an integer underflow */
        int result = data - 1;
        printIntLine(result);
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    fscanf (stdin, "%d", &data);
    {
        /* POTENTIAL FLAW: Subtracting 1 from data could cause an integer underflow */
        int result = data - 1;
        printIntLine(result);
    }
}

void CWE191_Integer_Underflow__int_fscanf_19_good()
{
    goodB2G();
    goodG2B();
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
    CWE191_Integer_Underflow__int_fscanf_19_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE191_Integer_Underflow__int_fscanf_19_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
