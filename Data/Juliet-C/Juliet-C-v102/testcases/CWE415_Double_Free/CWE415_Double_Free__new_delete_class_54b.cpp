/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE415_Double_Free__new_delete_class_54b.cpp
Label Definition File: CWE415_Double_Free__new_delete.label.xml
Template File: sources-sinks-54b.tmpl.cpp
*/
/*
 * @description
 * CWE: 415 Double Free
 * BadSource:  Allocate data using new and Deallocae data using delete
 * GoodSource: Allocate data using new
 * Sinks:
 *    GoodSink: do nothing
 *    BadSink : Deallocate data using delete
 * Flow Variant: 54 Data flow: data passed as an argument from one function through three others to a fifth; all five functions are in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE415_Double_Free__new_delete_class_54
{

#ifndef OMITBAD

/* bad function declaration */
void bad_sink_c(twointsclass * data);

void bad_sink_b(twointsclass * data)
{
    bad_sink_c(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void goodG2B_sink_c(twointsclass * data);

void goodG2B_sink_b(twointsclass * data)
{
    goodG2B_sink_c(data);
}

/* goodB2G uses the BadSource with the GoodSink */
void goodB2G_sink_c(twointsclass * data);

void goodB2G_sink_b(twointsclass * data)
{
    goodB2G_sink_c(data);
}

#endif /* OMITGOOD */

} // close namespace
