/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE415_Double_Free__new_delete_array_long_long_67a.cpp
Label Definition File: CWE415_Double_Free__new_delete_array.label.xml
Template File: sources-sinks-67a.tmpl.cpp
*/
/*
 * @description
 * CWE: 415 Double Free
 * BadSource:  Allocate data using new and Deallocae data using delete
 * GoodSource: Allocate data using new
 * Sinks:
 *    GoodSink: do nothing
 *    BadSink : Deallocate data using delete
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE415_Double_Free__new_delete_array_long_long_67
{

typedef struct _struct_type
{
    long long * a;
} struct_type;

#ifndef OMITBAD

/* bad function declaration */
void bad_sink(struct_type my_struct);

void bad()
{
    long long * data;
    struct_type my_struct;
    /* Initialize data */
    data = NULL;
    data = new long long[100];
    /* POTENTIAL FLAW: delete the array data in the source - the bad sink deletes the array data as well */
    delete [] data;
    my_struct.a = data;
    bad_sink(my_struct);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void goodG2B_sink(struct_type my_struct);

static void goodG2B()
{
    long long * data;
    struct_type my_struct;
    /* Initialize data */
    data = NULL;
    data = new long long[100];
    /* FIX: Do NOT delete the array data in the source - the bad sink deletes the array data */
    my_struct.a = data;
    goodG2B_sink(my_struct);
}

/* goodB2G uses the BadSource with the GoodSink */
void goodB2G_sink(struct_type my_struct);

static void goodB2G()
{
    long long * data;
    struct_type my_struct;
    /* Initialize data */
    data = NULL;
    data = new long long[100];
    /* POTENTIAL FLAW: delete the array data in the source - the bad sink deletes the array data as well */
    delete [] data;
    my_struct.a = data;
    goodB2G_sink(my_struct);
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

using namespace CWE415_Double_Free__new_delete_array_long_long_67; // so that we can use good and bad easily

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
