/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE190_Integer_Overflow__int_rand_add_18.c
Label Definition File: CWE190_Integer_Overflow__int.label.xml
Template File: sources-sinks-18.tmpl.c
*/
/*
 * @description
 * CWE: 190 Integer Overflow
 * BadSource: rand Set data to result of rand(), which may be zero
 * GoodSource: Small, non-zero
 * Sinks: add
 *    GoodSink: Ensure there is no overflow before performing the addition
 *    BadSink : Add 1 to data
 * Flow Variant: 18 Control flow: goto statements
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE190_Integer_Overflow__int_rand_add_18_bad()
{
    int data;
    /* Initialize data */
    data = -1;
    goto source;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    /* FIX: Use a small, non-zero value that will not cause an integer overflow in the sinks */
    data = 5;
source:
    data = RAND32();
    goto sink;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        int result = -1;
        /* FIX: Add a check to prevent an overflow from occurring */
        if (data < INT_MAX)
        {
            result = data + 1;
            printIntLine(result);
        }
        else
        {
            printLine("Input value is too large to perform arithmetic safely.");
        }
    }
sink:
    {
        /* POTENTIAL FLAW: Adding 1 to data could cause an integer overflow */
        int result = data + 1;
        printIntLine(result);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G() - use badsource and goodsink by reversing the blocks on the second goto statement */
static void goodB2G()
{
    int data;
    /* Initialize data */
    data = -1;
    goto source;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    /* FIX: Use a small, non-zero value that will not cause an integer overflow in the sinks */
    data = 5;
source:
    data = RAND32();
    goto sink;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        /* POTENTIAL FLAW: Adding 1 to data could cause an integer overflow */
        int result = data + 1;
        printIntLine(result);
    }
sink:
    {
        int result = -1;
        /* FIX: Add a check to prevent an overflow from occurring */
        if (data < INT_MAX)
        {
            result = data + 1;
            printIntLine(result);
        }
        else
        {
            printLine("Input value is too large to perform arithmetic safely.");
        }
    }
}

/* goodG2B() - use goodsource and badsink by reversing the blocks on the first goto statement */
static void goodG2B()
{
    int data;
    /* Initialize data */
    data = -1;
    goto source;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    data = RAND32();
source:
    /* FIX: Use a small, non-zero value that will not cause an integer overflow in the sinks */
    data = 5;
    goto sink;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        int result = -1;
        /* FIX: Add a check to prevent an overflow from occurring */
        if (data < INT_MAX)
        {
            result = data + 1;
            printIntLine(result);
        }
        else
        {
            printLine("Input value is too large to perform arithmetic safely.");
        }
    }
sink:
    {
        /* POTENTIAL FLAW: Adding 1 to data could cause an integer overflow */
        int result = data + 1;
        printIntLine(result);
    }
}

void CWE190_Integer_Overflow__int_rand_add_18_good()
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
    CWE190_Integer_Overflow__int_rand_add_18_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE190_Integer_Overflow__int_rand_add_18_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
