/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE415_Double_Free__new_delete_int_53a.cpp
Label Definition File: CWE415_Double_Free__new_delete.label.xml
Template File: sources-sinks-53a.tmpl.cpp
*/
/*
 * @description
 * CWE: 415 Double Free
 * BadSource:  Allocate data using new and Deallocae data using delete
 * GoodSource: Allocate data using new
 * Sinks:
 *    GoodSink: do nothing
 *    BadSink : Deallocate data using delete
 * Flow Variant: 53 Data flow: data passed as an argument from one function through two others to a fourth; all four functions are in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE415_Double_Free__new_delete_int_53
{

#ifndef OMITBAD

/* bad function declaration */
void bad_sink_b(int * data);

void bad()
{
    int * data;
    /* Initialize data */
    data = NULL;
    data = new int;
    /* POTENTIAL FLAW: delete data in the source - the bad sink deletes data as well */
    delete data;
    bad_sink_b(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void goodG2B_sink_b(int * data);

static void goodG2B()
{
    int * data;
    /* Initialize data */
    data = NULL;
    data = new int;
    /* FIX: Do NOT delete data in the source - the bad sink deletes data */
    goodG2B_sink_b(data);
}

/* goodB2G uses the BadSource with the GoodSink */
void goodB2G_sink_b(int * data);

static void goodB2G()
{
    int * data;
    /* Initialize data */
    data = NULL;
    data = new int;
    /* POTENTIAL FLAW: delete data in the source - the bad sink deletes data as well */
    delete data;
    goodB2G_sink_b(data);
}

void good()
{
    goodG2B();
    goodB2G();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

using namespace CWE415_Double_Free__new_delete_int_53; // so that we can use good and bad easily

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
