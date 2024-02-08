/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE415_Double_Free__new_delete_int_17.cpp
Label Definition File: CWE415_Double_Free__new_delete.label.xml
Template File: sources-sinks-17.tmpl.cpp
*/
/*
 * @description
 * CWE: 415 Double Free
 * BadSource:  Allocate data using new and Deallocae data using delete
 * GoodSource: Allocate data using new
 * Sinks:
 *    GoodSink: do nothing
 *    BadSink : Deallocate data using delete
 * Flow Variant: 17 Control flow: for loops
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE415_Double_Free__new_delete_int_17
{

#ifndef OMITBAD

void bad()
{
    int h,i,j,k;
    int * data;
    /* Initialize data */
    data = NULL;
    for(h = 0; h < 0; h++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        data = new int;
        /* FIX: Do NOT delete data in the source - the bad sink deletes data */
    }
    for(i = 0; i < 1; i++)
    {
        data = new int;
        /* POTENTIAL FLAW: delete data in the source - the bad sink deletes data as well */
        delete data;
    }
    for(j = 0; j < 1; j++)
    {
        /* POTENTIAL FLAW: Possibly deleting memory twice */
        delete data;
    }
    for(k = 0; k < 0; k++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* do nothing */
        /* FIX: Don't attempt to delete the memory */
        ; /* empty statement needed for some flow variants */
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G() - use badsource and goodsink by changing the conditions on the third and fourth for statements */
static void goodB2G()
{
    int h,i,j,k;
    int * data;
    /* Initialize data */
    data = NULL;
    for(h = 0; h < 0; h++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        data = new int;
        /* FIX: Do NOT delete data in the source - the bad sink deletes data */
    }
    for(i = 0; i < 1; i++)
    {
        data = new int;
        /* POTENTIAL FLAW: delete data in the source - the bad sink deletes data as well */
        delete data;
    }
    for(j = 0; j < 0; j++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Possibly deleting memory twice */
        delete data;
    }
    for(k = 0; k < 1; k++)
    {
        /* do nothing */
        /* FIX: Don't attempt to delete the memory */
        ; /* empty statement needed for some flow variants */
    }
}

/* goodG2B() - use goodsource and badsink by changing the conditions on the first and second for statements */
static void goodG2B()
{
    int h,i,j,k;
    int * data;
    /* Initialize data */
    data = NULL;
    for(h = 0; h < 1; h++)
    {
        data = new int;
        /* FIX: Do NOT delete data in the source - the bad sink deletes data */
    }
    for(i = 0; i < 0; i++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        data = new int;
        /* POTENTIAL FLAW: delete data in the source - the bad sink deletes data as well */
        delete data;
    }
    for(j = 0; j < 1; j++)
    {
        /* POTENTIAL FLAW: Possibly deleting memory twice */
        delete data;
    }
    for(k = 0; k < 0; k++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* do nothing */
        /* FIX: Don't attempt to delete the memory */
        ; /* empty statement needed for some flow variants */
    }
}

void good()
{
    goodB2G();
    goodG2B();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

using namespace CWE415_Double_Free__new_delete_int_17; // so that we can use good and bad easily

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
