	.file	"BotalovSerg_my_bot_examp_task_7_5_4_flatten.c"
	.text
	.globl	_TIG_IZ_FQdb_argv
	.bss
	.align 8
	.type	_TIG_IZ_FQdb_argv, @object
	.size	_TIG_IZ_FQdb_argv, 8
_TIG_IZ_FQdb_argv:
	.zero	8
	.globl	_TIG_IZ_FQdb_envp
	.align 8
	.type	_TIG_IZ_FQdb_envp, @object
	.size	_TIG_IZ_FQdb_envp, 8
_TIG_IZ_FQdb_envp:
	.zero	8
	.globl	_TIG_IZ_FQdb_argc
	.align 4
	.type	_TIG_IZ_FQdb_argc, @object
	.size	_TIG_IZ_FQdb_argc, 4
_TIG_IZ_FQdb_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"\320\257\320\267\321\213\320\272 Python"
	.align 8
.LC1:
	.string	"\320\235\320\265 \320\262\320\265\321\200\320\275\321\213\320\271 \320\277\321\203\320\275\320\272\321\202 \320\274\320\265\320\275\321\216"
.LC2:
	.string	"\320\257\320\267\321\213\320\272 Java"
.LC3:
	.string	"%d"
.LC4:
	.string	"\320\222\321\213\321\205\320\276\320\264"
.LC5:
	.string	"\320\257\320\267\321\213\320\272 \320\241\320\270"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_FQdb_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_FQdb_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_FQdb_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 100 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-FQdb--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_FQdb_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_FQdb_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_FQdb_envp(%rip)
	nop
	movq	$9, -16(%rbp)
.L24:
	cmpq	$12, -16(%rbp)
	ja	.L27
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L27-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L27-.L8
	.long	.L27-.L8
	.long	.L12-.L8
	.long	.L27-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L27-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L25
	jmp	.L26
.L11:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -16(%rbp)
	jmp	.L17
.L15:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -16(%rbp)
	jmp	.L17
.L13:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -16(%rbp)
	jmp	.L17
.L9:
	movl	-20(%rbp), %eax
	cmpl	$4, %eax
	je	.L18
	cmpl	$4, %eax
	jg	.L19
	cmpl	$3, %eax
	je	.L20
	cmpl	$3, %eax
	jg	.L19
	cmpl	$1, %eax
	je	.L21
	cmpl	$2, %eax
	je	.L22
	jmp	.L19
.L18:
	movq	$3, -16(%rbp)
	jmp	.L23
.L20:
	movq	$8, -16(%rbp)
	jmp	.L23
.L22:
	movq	$2, -16(%rbp)
	jmp	.L23
.L21:
	movq	$6, -16(%rbp)
	jmp	.L23
.L19:
	movq	$1, -16(%rbp)
	nop
.L23:
	jmp	.L17
.L10:
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$11, -16(%rbp)
	jmp	.L17
.L12:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -16(%rbp)
	jmp	.L17
.L14:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -16(%rbp)
	jmp	.L17
.L27:
	nop
.L17:
	jmp	.L24
.L26:
	call	__stack_chk_fail@PLT
.L25:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
