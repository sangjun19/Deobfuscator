	.file	"athiraomanakuttan_brototype_Week_2_5_flatten.c"
	.text
	.globl	_TIG_IZ_RiHR_envp
	.bss
	.align 8
	.type	_TIG_IZ_RiHR_envp, @object
	.size	_TIG_IZ_RiHR_envp, 8
_TIG_IZ_RiHR_envp:
	.zero	8
	.globl	_TIG_IZ_RiHR_argv
	.align 8
	.type	_TIG_IZ_RiHR_argv, @object
	.size	_TIG_IZ_RiHR_argv, 8
_TIG_IZ_RiHR_argv:
	.zero	8
	.globl	_TIG_IZ_RiHR_argc
	.align 4
	.type	_TIG_IZ_RiHR_argc, @object
	.size	_TIG_IZ_RiHR_argc, 4
_TIG_IZ_RiHR_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Please enter your mark"
.LC1:
	.string	"%f"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
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
	movq	$0, _TIG_IZ_RiHR_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_RiHR_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_RiHR_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 100 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-RiHR--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_RiHR_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_RiHR_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_RiHR_envp(%rip)
	nop
	movq	$1, -16(%rbp)
.L11:
	cmpq	$2, -16(%rbp)
	je	.L6
	cmpq	$2, -16(%rbp)
	ja	.L14
	cmpq	$0, -16(%rbp)
	je	.L15
	cmpq	$1, -16(%rbp)
	jne	.L14
	movq	$2, -16(%rbp)
	jmp	.L9
.L6:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-20(%rbp), %eax
	movd	%eax, %xmm0
	call	GradeCheck
	movq	$0, -16(%rbp)
	jmp	.L9
.L14:
	nop
.L9:
	jmp	.L11
.L15:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L13
	call	__stack_chk_fail@PLT
.L13:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.section	.rodata
.LC2:
	.string	"Grade D"
.LC3:
	.string	"Grade C"
.LC5:
	.string	"Sorry yu failed"
.LC6:
	.string	"Wrong input."
.LC7:
	.string	"Grade B"
.LC8:
	.string	"Grade A"
.LC9:
	.string	"Grade E"
	.text
	.globl	GradeCheck
	.type	GradeCheck, @function
GradeCheck:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movss	%xmm0, -36(%rbp)
	movq	$1, -16(%rbp)
.L49:
	cmpq	$24, -16(%rbp)
	ja	.L51
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L19(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L19(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L19:
	.long	.L51-.L19
	.long	.L33-.L19
	.long	.L51-.L19
	.long	.L32-.L19
	.long	.L31-.L19
	.long	.L51-.L19
	.long	.L30-.L19
	.long	.L29-.L19
	.long	.L51-.L19
	.long	.L51-.L19
	.long	.L51-.L19
	.long	.L28-.L19
	.long	.L27-.L19
	.long	.L26-.L19
	.long	.L25-.L19
	.long	.L51-.L19
	.long	.L51-.L19
	.long	.L24-.L19
	.long	.L23-.L19
	.long	.L22-.L19
	.long	.L21-.L19
	.long	.L51-.L19
	.long	.L20-.L19
	.long	.L51-.L19
	.long	.L18-.L19
	.text
.L23:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L34
.L31:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L34
.L25:
	movl	$0, %eax
	jmp	.L50
.L27:
	pxor	%xmm2, %xmm2
	cvtss2sd	-36(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	call	round@PLT
	movq	%xmm0, %rax
	movq	%rax, -8(%rbp)
	movsd	-8(%rbp), %xmm0
	movsd	.LC4(%rip), %xmm1
	divsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, -20(%rbp)
	movq	$22, -16(%rbp)
	jmp	.L34
.L33:
	movq	$12, -16(%rbp)
	jmp	.L34
.L32:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L34
.L18:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L34
.L28:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L34
.L26:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L34
.L22:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L34
.L24:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L34
.L30:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L34
.L20:
	cmpl	$9, -20(%rbp)
	ja	.L36
	movl	-20(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L38(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L38(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L38:
	.long	.L47-.L38
	.long	.L46-.L38
	.long	.L45-.L38
	.long	.L44-.L38
	.long	.L43-.L38
	.long	.L42-.L38
	.long	.L41-.L38
	.long	.L40-.L38
	.long	.L39-.L38
	.long	.L37-.L38
	.text
.L47:
	movq	$24, -16(%rbp)
	jmp	.L48
.L46:
	movq	$6, -16(%rbp)
	jmp	.L48
.L45:
	movq	$3, -16(%rbp)
	jmp	.L48
.L44:
	movq	$7, -16(%rbp)
	jmp	.L48
.L43:
	movq	$17, -16(%rbp)
	jmp	.L48
.L42:
	movq	$20, -16(%rbp)
	jmp	.L48
.L41:
	movq	$18, -16(%rbp)
	jmp	.L48
.L40:
	movq	$4, -16(%rbp)
	jmp	.L48
.L39:
	movq	$13, -16(%rbp)
	jmp	.L48
.L37:
	movq	$19, -16(%rbp)
	jmp	.L48
.L36:
	movq	$11, -16(%rbp)
	nop
.L48:
	jmp	.L34
.L29:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L34
.L21:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L34
.L51:
	nop
.L34:
	jmp	.L49
.L50:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	GradeCheck, .-GradeCheck
	.section	.rodata
	.align 8
.LC4:
	.long	0
	.long	1076101120
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
