/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE762_Mismatched_Memory_Management_Routines__delete_array_char_realloc_05.cpp
Label Definition File: CWE762_Mismatched_Memory_Management_Routines__delete_array.label.xml
Template File: sources-sinks-05.tmpl.cpp
*/
/*
 * @description
 * CWE: 762 Mismatched Memory Management Routines
 * BadSource: realloc Allocate data using realloc()
 * GoodSource: Allocate data using new []
 * Sinks:
 *    GoodSink: Deallocate data using free()
 *    BadSink : Deallocate data using delete []
 * Flow Variant: 05 Control flow: if(static_t) and if(static_f)
 *
 * */

#include "std_testcase.h"

/* The two variables below are not defined as "const", but are never
   assigned any other value, so a tool should be able to identify that
   reads of these will always return their initialized values. */
static int static_t = 1; /* true */
static int static_f = 0; /* false */

namespace CWE762_Mismatched_Memory_Management_Routines__delete_array_char_realloc_05
{

#ifndef OMITBAD

void bad()
{
    char * data;
    /* Initialize data*/
    data = NULL;
    if(static_t)
    {
        data = NULL;
        /* POTENTIAL FLAW: Allocate memory with a function that requires free() to free the memory */
        data = (char *)realloc(data, 100*sizeof(char));
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Allocate memory using new [] */
        data = new char[100];
    }
    if(static_t)
    {
        /* POTENTIAL FLAW: Deallocate memory using delete [] - the source memory allocation function may
         * require a call to free() to deallocate the memory */
        delete [] data;
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Free memory using free() */
        free(data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G1() - use badsource and goodsink by changing the second static_t to static_f */
static void goodB2G1()
{
    char * data;
    /* Initialize data*/
    data = NULL;
    if(static_t)
    {
        data = NULL;
        /* POTENTIAL FLAW: Allocate memory with a function that requires free() to free the memory */
        data = (char *)realloc(data, 100*sizeof(char));
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Allocate memory using new [] */
        data = new char[100];
    }
    if(static_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Deallocate memory using delete [] - the source memory allocation function may
         * require a call to free() to deallocate the memory */
        delete [] data;
    }
    else
    {
        /* FIX: Free memory using free() */
        free(data);
    }
}

/* goodB2G2() - use badsource and goodsink by reversing the blocks in the second if */
static void goodB2G2()
{
    char * data;
    /* Initialize data*/
    data = NULL;
    if(static_t)
    {
        data = NULL;
        /* POTENTIAL FLAW: Allocate memory with a function that requires free() to free the memory */
        data = (char *)realloc(data, 100*sizeof(char));
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Allocate memory using new [] */
        data = new char[100];
    }
    if(static_t)
    {
        /* FIX: Free memory using free() */
        free(data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Deallocate memory using delete [] - the source memory allocation function may
         * require a call to free() to deallocate the memory */
        delete [] data;
    }
}

/* goodG2B1() - use goodsource and badsink by changing the first static_t to static_f */
static void goodG2B1()
{
    char * data;
    /* Initialize data*/
    data = NULL;
    if(static_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        data = NULL;
        /* POTENTIAL FLAW: Allocate memory with a function that requires free() to free the memory */
        data = (char *)realloc(data, 100*sizeof(char));
    }
    else
    {
        /* FIX: Allocate memory using new [] */
        data = new char[100];
    }
    if(static_t)
    {
        /* POTENTIAL FLAW: Deallocate memory using delete [] - the source memory allocation function may
         * require a call to free() to deallocate the memory */
        delete [] data;
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Free memory using free() */
        free(data);
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the first if */
static void goodG2B2()
{
    char * data;
    /* Initialize data*/
    data = NULL;
    if(static_t)
    {
        /* FIX: Allocate memory using new [] */
        data = new char[100];
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        data = NULL;
        /* POTENTIAL FLAW: Allocate memory with a function that requires free() to free the memory */
        data = (char *)realloc(data, 100*sizeof(char));
    }
    if(static_t)
    {
        /* POTENTIAL FLAW: Deallocate memory using delete [] - the source memory allocation function may
         * require a call to free() to deallocate the memory */
        delete [] data;
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Free memory using free() */
        free(data);
    }
}

void good()
{
    goodB2G1();
    goodB2G2();
    goodG2B1();
    goodG2B2();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

using namespace CWE762_Mismatched_Memory_Management_Routines__delete_array_char_realloc_05; // so that we can use good and bad easily

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
