/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE415_Double_Free__new_delete_class_61a.cpp
Label Definition File: CWE415_Double_Free__new_delete.label.xml
Template File: sources-sinks-61a.tmpl.cpp
*/
/*
 * @description
 * CWE: 415 Double Free
 * BadSource:  Allocate data using new and Deallocae data using delete
 * GoodSource: Allocate data using new
 * Sinks:
 *    GoodSink: do nothing
 *    BadSink : Deallocate data using delete
 * Flow Variant: 61 Data flow: data returned from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE415_Double_Free__new_delete_class_61
{

#ifndef OMITBAD

/* bad function declaration */
twointsclass * bad_source(twointsclass * data);

void bad()
{
    twointsclass * data;
    /* Initialize data */
    data = NULL;
    data = bad_source(data);
    /* POTENTIAL FLAW: Possibly deleting memory twice */
    delete data;
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
twointsclass * goodG2B_source(twointsclass * data);

static void goodG2B()
{
    twointsclass * data;
    /* Initialize data */
    data = NULL;
    data = goodG2B_source(data);
    /* POTENTIAL FLAW: Possibly deleting memory twice */
    delete data;
}

/* goodB2G uses the BadSource with the GoodSink */
twointsclass * goodB2G_source(twointsclass * data);

static void goodB2G()
{
    twointsclass * data;
    /* Initialize data */
    data = NULL;
    data = goodB2G_source(data);
    /* do nothing */
    /* FIX: Don't attempt to delete the memory */
    ; /* empty statement needed for some flow variants */
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

using namespace CWE415_Double_Free__new_delete_class_61; // so that we can use good and bad easily

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
