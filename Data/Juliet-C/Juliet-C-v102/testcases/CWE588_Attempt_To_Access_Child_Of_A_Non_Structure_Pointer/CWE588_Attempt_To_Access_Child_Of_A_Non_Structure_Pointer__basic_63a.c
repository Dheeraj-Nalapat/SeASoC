/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_63a.c
Label Definition File: CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic.label.xml
Template File: sources-sink-63a.tmpl.c
*/
/*
 * @description
 * CWE: 588 Attempt to Access Child of a Non Structure Type
 * BadSource:  Void pointer to an int
 * GoodSource: Void pointer to a twoints struct
 * Sinks:
 *    BadSink : Print data
 * Flow Variant: 63 Data flow: pointer to data passed from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

/* bad function declaration */
void CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_63b_bad_sink(void * * data_ptr);

void CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_63_bad()
{
    void * data;
    twoints data_goodbuf;
    int data_badbuf = 100;
    data_goodbuf.a = 0;
    data_goodbuf.b = 0;
    /* FLAW: Set data to point to an int */
    data = &data_badbuf;
    CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_63b_bad_sink(&data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_63b_goodG2B_sink(void * * data);

static void goodG2B()
{
    void * data;
    twoints data_goodbuf;
    int data_badbuf = 100;
    data_goodbuf.a = 0;
    data_goodbuf.b = 0;
    /* FIX: Set data to point to a twoints struct */
    data = &data_goodbuf;
    CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_63b_goodG2B_sink(&data);
}

void CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_63_good()
{
    goodG2B();
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
    CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_63_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_63_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif