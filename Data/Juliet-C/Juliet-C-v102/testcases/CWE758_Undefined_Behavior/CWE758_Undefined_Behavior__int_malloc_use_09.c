/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE758_Undefined_Behavior__int_malloc_use_09.c
Label Definition File: CWE758_Undefined_Behavior.alloc.label.xml
Template File: point-flaw-09.tmpl.c
*/
/*
 * @description
 * CWE: 758 Undefined Behavior
 * Sinks: malloc_use
 *    GoodSink: Initialize then use data
 *    BadSink : Use data from malloc without initialization
 * Flow Variant: 09 Control flow: if(global_const_t) and if(global_const_f)
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE758_Undefined_Behavior__int_malloc_use_09_bad()
{
    if(global_const_t)
    {
        {
            int * pointer = (int *)malloc(sizeof(int));
            int data = *pointer; /* FLAW: the value pointed to by pointer is undefined */
            free(pointer);
            printIntLine(data);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int data;
            int * pointer = (int *)malloc(sizeof(int));
            data = 5;
            *pointer = data; /* FIX: Assign a value to the thing pointed to by pointer */
            {
                int data = *pointer;
                printIntLine(data);
            }
            free(pointer);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(global_const_f) instead of if(global_const_t) */
static void good1()
{
    if(global_const_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int * pointer = (int *)malloc(sizeof(int));
            int data = *pointer; /* FLAW: the value pointed to by pointer is undefined */
            free(pointer);
            printIntLine(data);
        }
    }
    else
    {
        {
            int data;
            int * pointer = (int *)malloc(sizeof(int));
            data = 5;
            *pointer = data; /* FIX: Assign a value to the thing pointed to by pointer */
            {
                int data = *pointer;
                printIntLine(data);
            }
            free(pointer);
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(global_const_t)
    {
        {
            int data;
            int * pointer = (int *)malloc(sizeof(int));
            data = 5;
            *pointer = data; /* FIX: Assign a value to the thing pointed to by pointer */
            {
                int data = *pointer;
                printIntLine(data);
            }
            free(pointer);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int * pointer = (int *)malloc(sizeof(int));
            int data = *pointer; /* FLAW: the value pointed to by pointer is undefined */
            free(pointer);
            printIntLine(data);
        }
    }
}

void CWE758_Undefined_Behavior__int_malloc_use_09_good()
{
    good1();
    good2();
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
    CWE758_Undefined_Behavior__int_malloc_use_09_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE758_Undefined_Behavior__int_malloc_use_09_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
