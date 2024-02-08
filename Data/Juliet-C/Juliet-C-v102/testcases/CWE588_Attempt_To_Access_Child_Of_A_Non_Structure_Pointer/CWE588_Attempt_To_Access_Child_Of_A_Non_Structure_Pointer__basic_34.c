/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_34.c
Label Definition File: CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic.label.xml
Template File: sources-sink-34.tmpl.c
*/
/*
 * @description
 * CWE: 588 Attempt to Access Child of a Non Structure Type
 * BadSource:  Void pointer to an int
 * GoodSource: Void pointer to a twoints struct
 * Sinks:
 *    BadSink : Print data
 * Flow Variant: 34 Data flow: use of a union containing two methods of accessing the same data (within the same function)
 *
 * */

#include "std_testcase.h"

typedef union
{
    void * a;
    void * b;
} CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_34_union_type;

#ifndef OMITBAD

void CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_34_bad()
{
    void * data;
    CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_34_union_type my_union;
    twoints data_goodbuf;
    int data_badbuf = 100;
    data_goodbuf.a = 0;
    data_goodbuf.b = 0;
    /* FLAW: Set data to point to an int */
    data = &data_badbuf;
    my_union.a = data;
    {
        void * data = my_union.b;
        /* POTENTIAL FLAW: Attempt to print a struct when data may be a non-struct data type */
        printStructLine((twoints *)data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    void * data;
    CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_34_union_type my_union;
    twoints data_goodbuf;
    int data_badbuf = 100;
    data_goodbuf.a = 0;
    data_goodbuf.b = 0;
    /* FIX: Set data to point to a twoints struct */
    data = &data_goodbuf;
    my_union.a = data;
    {
        void * data = my_union.b;
        /* POTENTIAL FLAW: Attempt to print a struct when data may be a non-struct data type */
        printStructLine((twoints *)data);
    }
}

void CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_34_good()
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
    CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_34_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE588_Attempt_To_Access_Child_Of_A_Non_Structure_Pointer__basic_34_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
