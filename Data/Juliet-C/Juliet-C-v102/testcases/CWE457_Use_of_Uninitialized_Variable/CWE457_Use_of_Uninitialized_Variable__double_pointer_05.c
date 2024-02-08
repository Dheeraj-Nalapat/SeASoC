/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE457_Use_of_Uninitialized_Variable__double_pointer_05.c
Label Definition File: CWE457_Use_of_Uninitialized_Variable.c.label.xml
Template File: sources-sinks-05.tmpl.c
*/
/*
 * @description
 * CWE: 457 Use of Uninitialized Variable
 * BadSource: no_init Don't initialize data
 * GoodSource: Initialize data
 * Sinks: use
 *    GoodSink: Initialize then use data
 *    BadSink : Use data
 * Flow Variant: 05 Control flow: if(static_t) and if(static_f)
 *
 * */

#include "std_testcase.h"

# include <wchar.h>

/* The two variables below are not defined as "const", but are never
   assigned any other value, so a tool should be able to identify that
   reads of these will always return their initialized values. */
static int static_t = 1; /* true */
static int static_f = 0; /* false */

#ifndef OMITBAD

void CWE457_Use_of_Uninitialized_Variable__double_pointer_05_bad()
{
    double * data;
    if(static_t)
    {
        /* Don't initialize data */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* initialize both the pointer and the data pointed to */
        data = (double *)malloc(sizeof(double));
        *data = 5.0;
    }
    if(static_t)
    {
        printDoubleLine(*data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* initialize both the pointer and the data pointed to */
        data = (double *)malloc(sizeof(double));
        *data = 5.0;
        printDoubleLine(*data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G1() - use badsource and goodsink by changing the second static_t to static_f */
static void goodB2G1()
{
    double * data;
    if(static_t)
    {
        /* Don't initialize data */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* initialize both the pointer and the data pointed to */
        data = (double *)malloc(sizeof(double));
        *data = 5.0;
    }
    if(static_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        printDoubleLine(*data);
    }
    else
    {
        /* initialize both the pointer and the data pointed to */
        data = (double *)malloc(sizeof(double));
        *data = 5.0;
        printDoubleLine(*data);
    }
}

/* goodB2G2() - use badsource and goodsink by reversing the blocks in the second if */
static void goodB2G2()
{
    double * data;
    if(static_t)
    {
        /* Don't initialize data */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* initialize both the pointer and the data pointed to */
        data = (double *)malloc(sizeof(double));
        *data = 5.0;
    }
    if(static_t)
    {
        /* initialize both the pointer and the data pointed to */
        data = (double *)malloc(sizeof(double));
        *data = 5.0;
        printDoubleLine(*data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        printDoubleLine(*data);
    }
}

/* goodG2B1() - use goodsource and badsink by changing the first static_t to static_f */
static void goodG2B1()
{
    double * data;
    if(static_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* Don't initialize data */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* initialize both the pointer and the data pointed to */
        data = (double *)malloc(sizeof(double));
        *data = 5.0;
    }
    if(static_t)
    {
        printDoubleLine(*data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* initialize both the pointer and the data pointed to */
        data = (double *)malloc(sizeof(double));
        *data = 5.0;
        printDoubleLine(*data);
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the first if */
static void goodG2B2()
{
    double * data;
    if(static_t)
    {
        /* initialize both the pointer and the data pointed to */
        data = (double *)malloc(sizeof(double));
        *data = 5.0;
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* Don't initialize data */
        ; /* empty statement needed for some flow variants */
    }
    if(static_t)
    {
        printDoubleLine(*data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* initialize both the pointer and the data pointed to */
        data = (double *)malloc(sizeof(double));
        *data = 5.0;
        printDoubleLine(*data);
    }
}

void CWE457_Use_of_Uninitialized_Variable__double_pointer_05_good()
{
    goodB2G1();
    goodB2G2();
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
    CWE457_Use_of_Uninitialized_Variable__double_pointer_05_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE457_Use_of_Uninitialized_Variable__double_pointer_05_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
