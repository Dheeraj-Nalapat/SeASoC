/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE758_Undefined_Behavior__int_pointer_alloca_use_18.c
Label Definition File: CWE758_Undefined_Behavior.alloc.label.xml
Template File: point-flaw-18.tmpl.c
*/
/*
 * @description
 * CWE: 758 Undefined Behavior
 * Sinks: alloca_use
 *    GoodSink: Initialize then use data
 *    BadSink : Use data from alloca without initialization
 * Flow Variant: 18 Control flow: goto statements
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE758_Undefined_Behavior__int_pointer_alloca_use_18_bad()
{
    goto sink;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        int * data;
        int * * pointer = (int * *)ALLOCA(sizeof(int *));
        /* initialize both the pointer and the data pointed to */
        data = (int *)malloc(sizeof(int));
        *data = 5;
        *pointer = data; /* FIX: Assign a value to the thing pointed to by pointer */
        {
            int * data = *pointer;
            printIntLine(*data);
        }
    }
sink:
    {
        int * * pointer = (int * *)ALLOCA(sizeof(int *));
        int * data = *pointer; /* FLAW: the value pointed to by pointer is undefined */
        printIntLine(*data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() reverses the blocks on the goto statement */
static void good1()
{
    goto sink;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        int * * pointer = (int * *)ALLOCA(sizeof(int *));
        int * data = *pointer; /* FLAW: the value pointed to by pointer is undefined */
        printIntLine(*data);
    }
sink:
    {
        int * data;
        int * * pointer = (int * *)ALLOCA(sizeof(int *));
        /* initialize both the pointer and the data pointed to */
        data = (int *)malloc(sizeof(int));
        *data = 5;
        *pointer = data; /* FIX: Assign a value to the thing pointed to by pointer */
        {
            int * data = *pointer;
            printIntLine(*data);
        }
    }
}

void CWE758_Undefined_Behavior__int_pointer_alloca_use_18_good()
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
    CWE758_Undefined_Behavior__int_pointer_alloca_use_18_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE758_Undefined_Behavior__int_pointer_alloca_use_18_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
