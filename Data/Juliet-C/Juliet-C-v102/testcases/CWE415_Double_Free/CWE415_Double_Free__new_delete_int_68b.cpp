/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE415_Double_Free__new_delete_int_68b.cpp
Label Definition File: CWE415_Double_Free__new_delete.label.xml
Template File: sources-sinks-68b.tmpl.cpp
*/
/*
 * @description
 * CWE: 415 Double Free
 * BadSource:  Allocate data using new and Deallocae data using delete
 * GoodSource: Allocate data using new
 * Sinks:
 *    GoodSink: do nothing
 *    BadSink : Deallocate data using delete
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE415_Double_Free__new_delete_int_68
{

extern int * bad_data;
extern int * goodG2B_data;
extern int * goodB2G_data;

#ifndef OMITBAD

void bad_sink()
{
    int * data = bad_data;
    /* POTENTIAL FLAW: Possibly deleting memory twice */
    delete data;
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void goodG2B_sink()
{
    int * data = goodG2B_data;
    /* POTENTIAL FLAW: Possibly deleting memory twice */
    delete data;
}

/* goodB2G uses the BadSource with the GoodSink */
void goodB2G_sink()
{
    int * data = goodB2G_data;
    /* do nothing */
    /* FIX: Don't attempt to delete the memory */
    ; /* empty statement needed for some flow variants */
}

#endif /* OMITGOOD */

} // close namespace
