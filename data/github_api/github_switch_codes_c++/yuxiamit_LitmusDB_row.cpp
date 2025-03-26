// Repository: yuxiamit/LitmusDB
// File: dbx1000_logging/storage/row.cpp

#include <mm_malloc.h>
#include "global.h"
#include "table.h"
#include "catalog.h"
#include "row.h"
#include "txn.h"
#include "row_lock.h"
#include "row_ts.h"
#include "row_mvcc.h"
#include "row_hekaton.h"
#include "row_occ.h"
#include "row_tictoc.h"
#include "row_silo.h"
#include "row_vll.h"
#include "row_detreserve.h"
#include "mem_alloc.h"
#include "manager.h"
#include "parallel_log.h"
#include <new>

RC 
row_t::init(table_t * host_table, uint64_t part_id, uint64_t row_id) {
	_row_id = row_id;
	_part_id = part_id;
	this->table = host_table;
	Catalog * schema = host_table->get_schema();
	int tuple_size = schema->get_tuple_size();
	data = (char *) _mm_malloc(sizeof(char) * tuple_size, ALIGN_SIZE);
#if LOG_ALGORITHM == LOG_PARALLEL
	_last_writer = (uint64_t)-1;
//	for (uint32_t i = 0; i < 4; i++)
//		_pred_vector[i] = 0;
//  #if LOG_TYPE == LOG_COMMAND && LOG_RECOVER
//	_version = NULL;
//	_num_versions = 0;
//	_min_ts = UINT64_MAX; 
//	_gc_time = 0;
//#endif
#endif
	return RCOK;
}
void 
row_t::init(int size) 
{
	data = (char *) _mm_malloc(size, ALIGN_SIZE);
}

RC 
row_t::switch_schema(table_t * host_table) {
	this->table = host_table;
	return RCOK;
}

void row_t::init_manager(row_t * row) {
#if CC_ALG == DL_DETECT || CC_ALG == NO_WAIT || CC_ALG == WAIT_DIE
    manager = (Row_lock *) mem_allocator.alloc(sizeof(Row_lock), _part_id);
	new (manager) Row_lock();
#elif CC_ALG == TIMESTAMP
    manager = (Row_ts *) mem_allocator.alloc(sizeof(Row_ts), _part_id);
#elif CC_ALG == MVCC
    manager = (Row_mvcc *) _mm_malloc(sizeof(Row_mvcc), ALIGN_SIZE);
#elif CC_ALG == HEKATON
    manager = (Row_hekaton *) _mm_malloc(sizeof(Row_hekaton), ALIGN_SIZE);
#elif CC_ALG == OCC
    manager = (Row_occ *) mem_allocator.alloc(sizeof(Row_occ), _part_id);
#elif CC_ALG == TICTOC
	manager = (Row_tictoc *) _mm_malloc(sizeof(Row_tictoc), ALIGN_SIZE);
#elif CC_ALG == SILO
	manager = (Row_silo *) _mm_malloc(sizeof(Row_silo), ALIGN_SIZE);
	new (manager) Row_silo();
	//assert((uint64_t)manager > 0x700000000000);
#elif CC_ALG == VLL
    manager = (Row_vll *) mem_allocator.alloc(sizeof(Row_vll), _part_id);
#endif

#if CC_ALG != HSTORE
	manager->init(this);
#endif
	_lti_addr = NULL;
}

table_t * row_t::get_table() { 
	return table; 
}

Catalog * row_t::get_schema() { 
	return get_table()->get_schema(); 
}

const char * row_t::get_table_name() { 
	return get_table()->get_table_name(); 
};
uint32_t 
row_t::get_tuple_size() {
	return get_schema()->get_tuple_size();
}

uint64_t row_t::get_field_cnt() { 
	return get_schema()->field_cnt;
}

void row_t::set_value(int id, void * ptr) {
	int datasize = get_schema()->get_field_size(id);
	int pos = get_schema()->get_field_index(id);
	//printf("datasize is %d, pos=%d, tuplesize=%d\n", datasize, pos,table->get_schema()->get_tuple_size());
	memcpy( &data[pos], ptr, datasize);
}

//ATTRIBUTE_NO_SANITIZE_ADDRESS
void row_t::set_value(int id, void * ptr, int size) {
	int pos = get_schema()->get_field_index(id);
	memcpy( &data[pos], ptr, size);
}

void row_t::set_value(const char * col_name, void * ptr) {
	uint64_t id = get_schema()->get_field_id(col_name);
	set_value(id, ptr);
}

SET_VALUE(uint64_t);
SET_VALUE(int64_t);
SET_VALUE(double);
SET_VALUE(UInt32);
SET_VALUE(SInt32);

GET_VALUE(uint64_t);
GET_VALUE(int64_t);
GET_VALUE(double);
GET_VALUE(UInt32);
GET_VALUE(SInt32);

char * row_t::get_value(int id) {
	int pos = get_schema()->get_field_index(id);
	return &data[pos];
}

char * row_t::get_value(char * col_name) {
	uint64_t pos = get_schema()->get_field_index(col_name);
	return &data[pos];
}

char * 
row_t::get_value(Catalog * schema, uint32_t col_id, char * data)
{
	return &data[ schema->get_field_index(col_id) ];
}

void 
row_t::set_value(Catalog * schema, uint32_t col_id, char * data, char * value)
{
	memcpy( &data[ schema->get_field_index(col_id) ],
			value,
			schema->get_field_size(col_id)
		  );
}

char * 
row_t::get_data(txn_man * txn, access_t type) 
{
/*#if LOG_ALGORITHM == LOG_PARALLEL && LOG_TYPE == LOG_COMMAND && LOG_RECOVER
	char * ret_data = data;
	Version * first_delete = NULL;
	// No need to latch the tuple.
	// Because no conflicts occur during recover, 
	if(type == RD) {
		Version * cur_version = _version;
		ts_t txn_ts = txn->get_recover_state()->commit_ts;
		
		while(cur_version && cur_version->ts > txn_ts) 
			cur_version = cur_version->next;
		if (cur_version)
			ret_data = cur_version->data;
	} else if(type == WR) {
		// TODO. reuse versions through the freequeue
		Version * new_version = (Version *) _mm_malloc(sizeof(Version) + get_tuple_size(), ALIGN_SIZE);
		new_version->data = (char *)((uint64_t)new_version + sizeof(Version));
		new_version->next = _version;
		new_version->txn_id = txn->get_recover_state()->txn_id;
		new_version->ts = txn->get_recover_state()->commit_ts;
//	uint64_t tt = get_sys_clock();
		if(_version) { 
			memcpy(new_version->data, _version->data, get_tuple_size());
			M_ASSERT(new_version->ts > _version->ts, 
					"new_version->ts=%ld (txn=%ld), _version->ts=%ld (txn=%ld) \n", 
					new_version->ts, new_version->txn_id, _version->ts, _version->txn_id);
			if (_version->next == NULL)
				_min_ts = new_version->ts;
//	INC_STATS(GET_THD_ID, debug9, get_sys_clock() - tt);
		} else {
			memcpy(new_version->data, this->data, get_tuple_size());
//	INC_STATS(GET_THD_ID, debug8, get_sys_clock() - tt);
		}
		_version = new_version;
		
		// If the oldest version of tuple is older than fence, garbage collect
		if(_min_ts < glob_manager->get_min_ts()) {
			_gc_time ++;
			uint64_t fence_ts = glob_manager->get_min_ts();
			Version * cur_version = _version;
            while(cur_version && cur_version->ts >= fence_ts) {
				if (cur_version->next)
					_min_ts = cur_version->ts;
                cur_version = cur_version->next;
			}
			
			assert(cur_version && cur_version->next);	
			first_delete = cur_version->next;
			cur_version->next = NULL;
			data = NULL;
		}
		ret_data = _version->data;
	} 
	if (first_delete) {
		// delete everyting after and including dirst_delete
		Version * cur_version = first_delete;
		while(cur_version) {
           	Version * del_version = cur_version;
			cur_version = cur_version->next;
            _mm_free(del_version);
		}
	}
	assert(ret_data);
	return ret_data;
#else*/
	return data; 
//#endif
}

char * 
row_t::get_data() 
{ 
	return data; 
}

void row_t::set_data(char * data, uint64_t size) { 
	memcpy(this->data, data, size);
}
// copy from the src to this
void row_t::copy(row_t * src) {
	set_data(src->get_data(), src->get_tuple_size());
}

void row_t::copy(char * src) {
	set_data(src, get_tuple_size());
}

void row_t::free_row() {
	free(data);
}

//RC row_t::get_row(access_t type, txn_man * txn, row_t *& row) {
RC row_t::get_row(access_t type, txn_man * txn, char *& data) {
	RC rc = RCOK;
	//uint64_t starttime = get_sys_clock();
#if CC_ALG == WAIT_DIE || CC_ALG == NO_WAIT || CC_ALG == DL_DETECT
	//uint64_t thd_id = txn->get_thd_id();
	lock_t lt = (type == RD || type == SCAN)? LOCK_SH_T : LOCK_EX_T;
#if CC_ALG == DL_DETECT
	uint64_t * txnids;
	int txncnt; 
	rc = this->manager->lock_get(lt, txn, txnids, txncnt);	
#else
	rc = this->manager->lock_get(lt, txn);
#endif
	//uint64_t afterlockget = get_sys_clock();
	//INC_INT_STATS(time_debug6, afterlockget - starttime);
	// TODO: do we implement writes?
	// copy(data);
	if (rc == RCOK) {
		data = this->get_data(); //row = this;
	} else if (rc == Abort) {} 
	else if (rc == WAIT) {
		ASSERT(CC_ALG == WAIT_DIE || CC_ALG == DL_DETECT);
		/*uint64_t starttime = get_sys_clock();
#if CC_ALG == DL_DETECT	
		bool dep_added = false;
#endif
		uint64_t endtime;
		txn->lock_abort = false;
		INC_STATS(txn->get_thd_id(), wait_cnt, 1);
		while (!txn->lock_ready && !txn->lock_abort) 
		{
#if CC_ALG == WAIT_DIE 
			continue;
#elif CC_ALG == DL_DETECT	
			uint64_t last_detect = starttime;
			uint64_t last_try = starttime;

			uint64_t now = get_sys_clock();
			if (now - starttime > g_timeout ) {
				txn->lock_abort = true;
				break;
			}
			if (g_no_dl) {
				PAUSE
				continue;
			}
			int ok = 0;
			if ((now - last_detect > g_dl_loop_detect) && (now - last_try > DL_LOOP_TRIAL)) {
				if (!dep_added) {
					ok = dl_detector.add_dep(txn->get_txn_id(), txnids, txncnt, txn->row_cnt);
					if (ok == 0)
						dep_added = true;
					else if (ok == 16)
						last_try = now;
				}
				if (dep_added) {
					ok = dl_detector.detect_cycle(txn->get_txn_id());
					if (ok == 16)  // failed to lock the deadlock detector
						last_try = now;
					else if (ok == 0) 
						last_detect = now;
					else if (ok == 1) {
						last_detect = now;
					}
				}
			} else 
				PAUSE
#endif
		}
		if (txn->lock_ready) 
			rc = RCOK;
		else if (txn->lock_abort) { 
			rc = Abort;
			return_row(type, txn, NULL);
		}
		endtime = get_sys_clock();
		INC_TMP_STATS(thd_id, time_wait, endtime - starttime);
		row = this;*/
	}
	//INC_INT_STATS(time_debug7, get_sys_clock() - afterlockget);
	return rc;
/*
#elif CC_ALG == TIMESTAMP || CC_ALG == MVCC || CC_ALG == HEKATON 
	uint64_t thd_id = txn->get_thd_id();
	// For TIMESTAMP RD, a new copy of the row will be returned.
	// for MVCC RD, the version will be returned instead of a copy
	// So for MVCC RD-WR, the version should be explicitly copied.
	//row_t * newr = NULL;
  #if CC_ALG == TIMESTAMP
	// TODO. should not call malloc for each row read. Only need to call malloc once 
	// before simulation starts, like TicToc and Silo.
	txn->cur_row = (row_t *) mem_allocator.alloc(sizeof(row_t), this->get_part_id());
	txn->cur_row->init(get_table(), this->get_part_id());
  #endif

	// TODO need to initialize the table/catalog information.
	TsType ts_type = (type == RD)? R_REQ : P_REQ; 
	rc = this->manager->access(txn, ts_type, row);
	if (rc == RCOK ) {
		row = txn->cur_row;
	} else if (rc == WAIT) {
		uint64_t t1 = get_sys_clock();
		while (!txn->ts_ready)
			PAUSE
		uint64_t t2 = get_sys_clock();
		INC_TMP_STATS(thd_id, time_wait, t2 - t1);
		row = txn->cur_row;
	}
	if (rc != Abort) {
		row->table = get_table();
		assert(row->get_schema() == this->get_schema());
	}
	return rc;
#elif CC_ALG == OCC
	// OCC always make a local copy regardless of read or write
	txn->cur_row = (row_t *) mem_allocator.alloc(sizeof(row_t), get_part_id());
	txn->cur_row->init(get_table(), get_part_id());
	rc = this->manager->access(txn, R_REQ);
	row = txn->cur_row;
	return rc;
*/
#elif CC_ALG == TICTOC || CC_ALG == SILO
	// like OCC, tictoc also makes a local copy for each read/write
	//row->table = get_table();
	TsType ts_type = (type == RD)? R_REQ : P_REQ; 
	assert((uint64_t)this->manager > 0x700000000000);
	rc = this->manager->access(txn, ts_type, data);
	return rc;
#elif CC_ALG == HSTORE || CC_ALG == VLL
	row = this;
	return rc;
#elif CC_ALG == DETRESERVE
	// return the data directly.
	if(type==RD)
		memcpy(data, this->data, get_tuple_size());
	return RCOK;
#else
	assert(false);
#endif
	return rc;
}

// the "row" is the row read out in get_row(). 
// For locking based CC_ALG, the "row" is the same as "this". 
// For timestamp based CC_ALG, the "row" != "this", and the "row" must be freed.
// For MVCC, the row will simply serve as a version. The version will be 
// delete during history cleanup.
// For TIMESTAMP, the row will be explicity deleted at the end of access().
// (cf. row_ts.cpp)
void row_t::return_row(access_t type, txn_man * txn, char * data) {	
//uint64_t starttime = get_sys_clock();
#if CC_ALG == WAIT_DIE || CC_ALG == NO_WAIT || CC_ALG == DL_DETECT
	//assert (row == NULL || row == this || type == XP);
	if (ROLL_BACK && type == XP) {// recover from previous writes.
		//this->copy(row);
		copy(data);  // if rollback
	}
	this->manager->lock_release(txn);
#elif CC_ALG == TIMESTAMP || CC_ALG == MVCC 
	// for RD or SCAN or XP, the row should be deleted.
	// because all WR should be companied by a RD
	// for MVCC RD, the row is not copied, so no need to free. 
  #if CC_ALG == TIMESTAMP
	if (type == RD || type == SCAN) {
		row->free_row();
		mem_allocator.free(row, sizeof(row_t));
	}
  #endif
	if (type == XP) {
		this->manager->access(txn, XP_REQ, row);
	} else if (type == WR) {
		assert (type == WR && row != NULL);
		assert (row->get_schema() == this->get_schema());
		RC rc = this->manager->access(txn, W_REQ, row);
		assert(rc == RCOK);
	}
#elif CC_ALG == OCC
	assert (row != NULL);
	if (type == WR)
		manager->write( row, txn->end_ts );
	row->free_row();
	mem_allocator.free(row, sizeof(row_t));
	return;
#elif CC_ALG == TICTOC || CC_ALG == SILO
	assert (data != NULL);
	return;
#elif CC_ALG == HSTORE || CC_ALG == VLL
	return;
#else 
	assert(false);
#endif
//INC_INT_STATS(time_debug1, get_sys_clock() - starttime);
}

