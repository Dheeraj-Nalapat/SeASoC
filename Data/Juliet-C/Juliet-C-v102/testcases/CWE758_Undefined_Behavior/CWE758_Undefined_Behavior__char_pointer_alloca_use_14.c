/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE758_Undefined_Behavior__char_pointer_alloca_use_14.c
Label Definition File: CWE758_Undefined_Behavior.alloc.label.xml
Template File: point-flaw-14.tmpl.c
*/
/*
 * @description
 * CWE: 758 Undefined Behavior
 * Sinks: alloca_use
 *    GoodSink: Initialize then use data
 *    BadSink : Use data from alloca without initialization
 * Flow Variant: 14 Control flow: if(global_five==5) and if(global_five!=5)
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE758_Undefined_Behavior__char_pointer_alloca_use_14_bad()
{
    if(global_five==5)
    {
        {
            char * * pointer = (char * *)ALLOCA(sizeof(char *));
            char * data = *pointer; /* FLAW: the value pointed to by pointer is undefined */
            printLine(data);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            char * data;
            char * * pointer = (char * *)ALLOCA(sizeof(char *));
            data = "string";
            *pointer = data; /* FIX: Assign a value to the thing pointed to by pointer */
            {
                char * data = *pointer;
                printLine(data);
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(global_five!=5) instead of if(global_five==5) */
static void good1()
{
    if(global_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            char * * pointer = (char * *)ALLOCA(sizeof(char *));
            char * data = *pointer; /* FLAW: the value pointed to by pointer is undefined */
            printLine(data);
        }
    }
    else
    {
        {
            char * data;
            char * * pointer = (char * *)ALLOCA(sizeof(char *));
            data = "string";
            *pointer = data; /* FIX: Assign a value to the thing pointed to by pointer */
            {
                char * data = *pointer;
                printLine(data);
            }
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(global_five==5)
    {
        {
            char * data;
            char * * pointer = (char * *)ALLOCA(sizeof(char *));
            data = "string";
            *pointer = data; /* FIX: Assign a value to the thing pointed to by pointer */
            {
                char * data = *pointer;
                printLine(data);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            char * * pointer = (char * *)ALLOCA(sizeof(char *));
            char * data = *pointer; /* FLAW: the value pointed to by pointer is undefined */
            printLine(data);
        }
    }
}

void CWE758_Undefined_Behavior__char_pointer_alloca_use_14_good()
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
    CWE758_Undefined_Behavior__char_pointer_alloca_use_14_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE758_Undefined_Behavior__char_pointer_alloca_use_14_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif