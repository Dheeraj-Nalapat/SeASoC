/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE131_Incorrect_Calculation_Of_Buffer_Size__memcpy_08.c
Label Definition File: CWE131_Incorrect_Calculation_Of_Buffer_Size.missing_sizeof.label.xml
Template File: sources-sink-08.tmpl.c
*/
/*
 * @description
 * CWE: 131 Incorrect Calculation of Buffer Size
 * BadSource:  Allocate memory without using sizeof(int)
 * GoodSource: Allocate memory using sizeof(int)
 * Sink: memcpy
 *    BadSink : Copy array to data using memcpy()
 * Flow Variant: 08 Control flow: if(static_returns_t()) and if(static_returns_f())
 *
 * */

#include "std_testcase.h"

/* The two function below always return the same value, so a tool
   should be able to identify that calls to the functions will always
   return a fixed value. */
static int static_returns_t()
{
    return 1;
}

static int static_returns_f()
{
    return 0;
}

#ifndef OMITBAD

void CWE131_Incorrect_Calculation_Of_Buffer_Size__memcpy_08_bad()
{
    int * data;
    data = NULL;
    if(static_returns_t())
    {
        /* FLAW: Allocate memory without using sizeof(int *) */
        data = (int *)malloc(10);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Allocate memory using sizeof(int *) */
        data = (int *)malloc(10*sizeof(int));
    }
    {
        int data_src[10] = {0};
        /* POTENTIAL FLAW: Possible buffer overflow if data was not allocated correctly in the source */
        memcpy(data, data_src, 10*sizeof(int));
        printIntLine(data[0]);
        free(data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B1() - use goodsource and badsink by changing the static_returns_t() to static_returns_f() */
static void goodG2B1()
{
    int * data;
    data = NULL;
    if(static_returns_f())
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: Allocate memory without using sizeof(int *) */
        data = (int *)malloc(10);
    }
    else
    {
        /* FIX: Allocate memory using sizeof(int *) */
        data = (int *)malloc(10*sizeof(int));
    }
    {
        int data_src[10] = {0};
        /* POTENTIAL FLAW: Possible buffer overflow if data was not allocated correctly in the source */
        memcpy(data, data_src, 10*sizeof(int));
        printIntLine(data[0]);
        free(data);
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the if statement */
static void goodG2B2()
{
    int * data;
    data = NULL;
    if(static_returns_t())
    {
        /* FIX: Allocate memory using sizeof(int *) */
        data = (int *)malloc(10*sizeof(int));
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: Allocate memory without using sizeof(int *) */
        data = (int *)malloc(10);
    }
    {
        int data_src[10] = {0};
        /* POTENTIAL FLAW: Possible buffer overflow if data was not allocated correctly in the source */
        memcpy(data, data_src, 10*sizeof(int));
        printIntLine(data[0]);
        free(data);
    }
}

void CWE131_Incorrect_Calculation_Of_Buffer_Size__memcpy_08_good()
{
    goodG2B1();
    goodG2B2();
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
    CWE131_Incorrect_Calculation_Of_Buffer_Size__memcpy_08_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE131_Incorrect_Calculation_Of_Buffer_Size__memcpy_08_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
